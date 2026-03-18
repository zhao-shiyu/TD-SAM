import nibabel as nib
from medpy import metric
import numpy as np
import glob
import pandas as pd
import numpy

# path = '/mnt/sdb/zhaoshiyu/project_results/MA_SAM_endovis18/result_20231224/epoch_399/'
# path = '/mnt/sdb/zhaoshiyu/project_results/timesam_1_endovis18/results_20240331_2/epoch_299/'
# path = '/mnt/sdb/zhaoshiyu/project_results/MA_SAM_finetuning_endovis18/result_20240129_2/epoch_399/'
# path = '/mnt/sdb/zhaoshiyu/project_results/TDSAM_endovis17/result_fold3_20241205/epoch_299_endovis17/'
# path = '/mnt/sdb/zhaoshiyu/project_results/TDSAM_endovis18_8/result_20241121/epoch_299_nal_endovis/'
path = '/mnt/sdb/zhaoshiyu/project_results/TDSAM_endovis17/result_fold3_20250106/epoch_299/'

pred_path = sorted(glob.glob(path + '002*_pred.nii.gz'))
gt_path = sorted(glob.glob(path + '002*_gt.nii.gz'))

print(path)

mean_dice = []

def calculate_metric_percase(pred, gt):
    # area1 = pred.sum()
    # area2 = gt.sum()
    # inter = ((pred == 1) & (gt == 1) & (pred + gt == 1)).sum() 

    intersection = numpy.count_nonzero(pred & gt)
    size_i1 = numpy.count_nonzero(pred)
    size_i2 = numpy.count_nonzero(gt)

    # if area1 == 0 and area2 == 0:
        # return 0  # 如果预测和真实标签都是空，则IoU为1
        # return 1.0  # 如果预测和真实标签都是空，则IoU为1
    # mask_iou = 2 * inter / (area1 + area2 + 1e-20)

    try:
        dice = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dice = 0.0

    return dice

for i in range(len(pred_path)):
    pred = nib.load(pred_path[i])
    gt = nib.load(gt_path[i])
    pred = pred.get_fdata()
    gt = gt.get_fdata()

    pred = np.transpose(pred, (2, 0, 1))
    gt = np.transpose(gt, (2, 0, 1))

    classes = 11
    metric_num = []
    metric_list = []

    arr = []

    for slice in range(pred.shape[0]):

        prediction = pred[slice]
        label = gt[slice]

        for i in range(1, classes + 1):
            metric_num.append(calculate_metric_percase(prediction == i, label == i))
            # metric_value = calculate_metric_percase(prediction == i, label == i)
            # if metric_value != 1:  # 排除值为1的情况
            #     metric_num.append(metric_value)

        for j in range(len(metric_num)):
            # if metric_num[j] > 1e-20:
            if metric_num[j] != 0:
                arr.append(metric_num[j])
                
        # print("metric_num:", metric_num)
        # print("arr:", arr)

        if len(arr) > 0:
            metric_list.append(np.mean(arr))
        
        metric_num = []
        arr = []


    res = np.mean(metric_list)
    print(res)
    mean_dice.append(res)


print("mean dice is : ", np.mean(mean_dice))
print('dice finished')