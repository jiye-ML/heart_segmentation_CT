# encoding=utf8
'''
查看和显示nii文件
'''

import matplotlib

matplotlib.use('TkAgg')
example_filename = r'F:\train_data2\MM-WHS\ct_train2'
import numpy as np
import os  # 遍历文件夹
import nibabel as nib
import imageio  # 转换成图像


def nii_to_image(niifile):
    filenames = os.listdir(filepath)  # 读取nii文件夹
    slice_trans = []

    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  # 读取nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii', '')  # 去掉nii的后缀名
        img_f_path = os.path.join(imgfile, fname)
        # 创建nii对应的图像的文件夹
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)  # 新建文件夹

        # 开始转换为图像
        (x, y, z) = img.shape
        for i in range(z):  # z是图像的序列
            silce = img_fdata[i, :, :]  # 选择哪个方向的切片都可以
            imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), silce)
            # 保存图像


if __name__ == '__main__':
    filepath = example_filename
    imgfile = r"F:\train_data2\MM-WHS\train1-img"
    nii_to_image(filepath)
