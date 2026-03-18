import nibabel as nib
from medpy import metric
import numpy as np
import glob
import numpy

# path = '/mnt/sdb/zhaoshiyu/project_results/MA_SAM_endovis18/result_20231224/epoch_399/'
# path = '/mnt/sdb/zhaoshiyu/project_results/timesam_1_endovis18/results_20240331_2/epoch_399/'
# path = '/mnt/sdb/zhaoshiyu/project_results/MA_SAM_finetuning_endovis18/result_20240129_2/epoch_399/'
# path = '/mnt/sdb/zhaoshiyu/project_results/TDSAM_endovis17/result_fold3_20241205/epoch_299_endovis17/'
path = '/mnt/sdb/zhaoshiyu/project_results/TDSAM_endovis17/result_fold3_20250106/epoch_299/'

pred_path = sorted(glob.glob(path + '002*_pred.nii.gz'))
gt_path = sorted(glob.glob(path + '002*_gt.nii.gz'))

print(path)

mean_iou = []

def mask_iou(mask1, mask2):
    # area1 = mask1.sum()
    # area2 = mask2.sum()
    # inter = ((mask1 == 1) & (mask2 == 1) & (mask1+mask2 == 1)).sum()
    # mask_iou = inter / (area1+area2-inter)

    intersection = numpy.count_nonzero(mask1 & mask2)
    size_i1 = numpy.count_nonzero(mask1)
    size_i2 = numpy.count_nonzero(mask2)

    try:
        mask_iou = intersection / (size_i1+size_i2-intersection)
    except ZeroDivisionError:
        mask_iou = 0.0

    return mask_iou

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
            metric_num.append(mask_iou(prediction == i, label == i))

        for j in range(len(metric_num)):
            if metric_num[j] != 0:
                arr.append(metric_num[j])
        if len(arr) > 0:
            metric_list.append(np.mean(arr))

        metric_num = []
        arr = []

    res = np.mean(metric_list)
    print(res)
    mean_iou.append(res)
    
print("mean iou is : ", np.mean(mean_iou))
print('iou finished')



# def mask_iou_2(mask1, mask2):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     pred[pred == 0] = 0
#     gt[gt == 0] = 0
#     area1 = mask1.sum()
#     area2 = mask2.sum()
#     inter = ((mask1+mask2) == 2).sum()
#     mask_iou = inter / (area1+area2-inter + 1e-8)
#     return mask_iou

'''

# compute_mIoU('', '', '', 12, [])

'''

'''
# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    a = a.astype(int)
    b = b.astype(int)
    return np.bincount(n * a[k] + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(pred.shape[0]): 

        prediction = np.array(pred[ind])
        label = np.array(gt[ind]) 
        #------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), prediction.flatten(),num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        # if ind > 0 and ind % 10 == 0:  
        #     print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, 250,
        #                                             100 * np.nanmean(per_class_iu(hist)),
        #                                             100 * np.nanmean(per_class_PA(hist))))
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    # for ind_class in range(num_classes):
    #     print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(round(mPA[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))  
    return mIoUs

'''