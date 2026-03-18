import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry

from icecream import ic
import pandas as pd
import pickle
from datetime import datetime
from einops import repeat
from scipy.ndimage import zoom
from utils import calculate_metric_percase
import nibabel as nib

# from utils import Process_label

gpu_ids = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

HU_min, HU_max = -200, 250
data_mean = 50.21997497685108
data_std = 68.47153712416372


def calculate_metric_percase(pred, gt):

    intersection = np.count_nonzero(pred & gt)
    size_i1 = np.count_nonzero(pred)
    size_i2 = np.count_nonzero(gt)

    try:
        dice = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dice = 0.0

    return dice


def test_single_volume(image, label, net, classes, multimask_output, patch_size=[512, 512], test_save_path=None, case=None):

    x, y = image.shape[0], image.shape[1]
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
    
    inputs = torch.from_numpy(image).unsqueeze(0).float().cuda()
    inputs = torch.permute(inputs, (0, -1, 3, 1, 2))
    
    label = torch.from_numpy(label).float().cuda()
    label = torch.permute(label, (2, 0, 1)) 

    classes = 11
    # label_batch = Process_label(label, prompt)

    net.eval()
    with torch.no_grad():
        outputs = net(inputs, multimask_output, patch_size[0])
        output_masks = outputs['masks']

        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)
        pred = out[2]
        gt = label[2]

        pred = pred.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)


        # num = pred.shape[0]
        dice_list = []
        metric_num = []
        arr = []

        # for i in range(num):
        for i in range(1, classes + 1):
            metric_num.append(calculate_metric_percase(pred == i, gt == i))
        for j in range(len(metric_num)):
            # if metric_num[j] > 1e-20:
            if metric_num[j] != 0:
                arr.append(metric_num[j])
        if len(arr) > 0:
            dice_list.append(np.mean(arr))

        mean_dice = np.mean(dice_list)

    return mean_dice

def inference(args, multimask_output, model, test_save_path=None):
    data_fd_list = pd.read_csv(args.data_path+'/test.csv')
    data_fd_list = data_fd_list["image_pth"]
    data_fd_list = [data_fd.split("/")[-3] for data_fd in data_fd_list]
    data_fd_list = list(set(data_fd_list))
    data_fd_list.sort()
    
    model.eval()
    dice_all_list = []

    for data_fd in tqdm(data_fd_list):
        image_file_list = os.listdir(args.data_path+'/'+data_fd + '/images')
        image_file_list.sort()

        dice_list = []

        for image_file in image_file_list:

            with open(args.data_path+'/'+data_fd + '/images/'+image_file, 'rb') as file:
                image_arr = pickle.load(file)
            with open(args.data_path+'/'+data_fd + '/masks/'+image_file.replace("2Dimage", "2Dmask"), 'rb') as file:
                mask_arr = pickle.load(file)


            image_arr = np.clip(image_arr, HU_min, HU_max)
            image_arr = (image_arr-HU_min)/(HU_max-HU_min)*255.0
            image_arr = np.float32(image_arr)
            image_arr = (image_arr - data_mean) / data_std
            image_arr = (image_arr-image_arr.min())/(image_arr.max()-image_arr.min()+0.00000001)
            mask_arr = np.float32(mask_arr)

            case_name = data_fd

            dice_i = test_single_volume(image_arr, mask_arr, model, classes=args.num_classes, multimask_output=multimask_output,
                                            patch_size=[args.img_size, args.img_size],
                                            test_save_path=test_save_path, case=case_name)

            print(data_fd, image_file, " dice is: ", dice_i)
            dice_list.append(dice_i)            
        
        dice_all_list.append(np.mean(dice_list))
        dice_list = []
        print('dice_all_list: ', dice_all_list)

    performance = np.mean(dice_all_list)
    
    print('Testing performance mean_dice : %f ' % (performance))
    print("Testing Finished!")
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt_ckpt', type=str, default='/mnt/sdb/zhaoshiyu/project_results/timesam_1_endovis18/results_20240723_vith_cfa/epoch_199.pth', help='The checkpoint after adaptation')
    parser.add_argument('--data_path', type=str, default='/mnt/sdb/zhaoshiyu/endovis18/2D_all_5slice/')
    
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/home/zhaoshiyu/Timesam_1/sam_vit_h_4b8939.pth', help='Pretrained checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_h', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=32, help='Rank for FacT adaptation')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--module', type=str, default='sam_fact_tt_image_encoder')

    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    args.output_dir = args.adapt_ckpt[:-4]
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0., 0., 0.],
                                                                pixel_std=[1., 1., 1.])
    
    pkg = import_module(args.module)
    net = pkg.Fact_tt_Sam(sam, args.rank, s=args.scale).cuda()

    assert args.adapt_ckpt is not None
    net.load_parameters(args.adapt_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    
    if not os.path.exists('./testing_log'):
        os.mkdir('./testing_log')
    logging.basicConfig(filename= './testing_log/' + args.adapt_ckpt.split('/')[-3] + '_log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = args.output_dir
    else:
        test_save_path = None
    inference(args, multimask_output, net, test_save_path)
