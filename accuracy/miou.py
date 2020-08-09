import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        union = (
            np.sum(self.confusionMatrix, axis=1) +
            np.sum(self.confusionMatrix, axis=0) -
            np.diag(self.confusionMatrix)
        )
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        imgLabel[imgLabel != 0] = 1
        imgPredict[imgPredict != 0] = 1
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)

        # 混淆均镇的某一行 + 预测到的某一列
        label = self.numClass * imgLabel[mask] + imgPredict[mask]

        # 判断每种label出现此处， 比如 0.1,2,3 ， 如果是二分类，则
        #                                           0,1为label=0 的预测为[0, 1] 的数目
        #                                           2,3为label=1的预测为[0, 1] 的数目
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


import glob
from PIL import Image

if __name__ == '__main__':

    all_file = glob.glob("../img_real/*.png")

    metric = SegmentationMetric(2)  # 14+1
    for file in all_file:
        print(file)
        if file.endswith("59.png"):
            print("sdfsfd")
        img = Image.open(file)
        predictImg = file.replace("img_real", "img_out")
        predictImg = Image.open(predictImg)
        imgPredict = np.array(predictImg)[:, :, 0]  # 可直接换成预测图片
        orininal_h1 = np.array(imgPredict).shape[0]
        orininal_w1 = np.array(imgPredict).shape[1]
        img = img.resize((orininal_w1, orininal_h1))
        imgLabel = np.array(img)  # 可直接换成标注图片
        metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print('pa is : %f' % pa)
    print('cpa is :')  # 列表
    print(cpa)
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % mIoU)
