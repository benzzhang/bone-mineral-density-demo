'''
Author: Peng Bo
Date: 2022-05-14
Description: Vertebra Segmentation dataset

'''
from heapq import merge
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch
import cv2
import numpy as np
import nibabel
import os
import sys
import imageio
import time
import random
import itertools
from utils.progress_bar import format_time

sys.path.append('../')


__all__ = ['VertebraDataset_2D', 'Infer_VertebraDataset_2D']

class Infer_VertebraDataset_2D(Dataset):

    def __init__(self, img_list, subset, prefix='data/', flag_linear=True):
        begin_time = time.time()

        self.img_list = [l.strip() for l in open(img_list).readlines()]
        self.subset = subset
        self.prefix = prefix
        self.flag_linear = flag_linear
        self.merge_img_data, self.sagittal_len, self.transverse_len, self.img_name_list = self.__readAllImgData__()
        print('loading %s data %s cost time: %s' % (self.subset, self.merge_img_data.shape, format_time(time.time()-begin_time)))

    def __readAllImgData__(self):
        img_data_list = []
        sagittal_len = []
        transverse_len = []
        img_name_list = []
        for index in range(len(self.img_list)):
            img_name = self.img_list[index].strip()
            img_path = os.path.join(self.prefix, img_name)
            img = nibabel.load(img_path).get_fdata()

            img_new = np.zeros((512, 512, 512))
            for i in range(img.shape[0]):
                slice = img[i, :, :]
                bg = min(slice.flatten())
                slice_new = np.pad(slice, ((0, 0), (0, 512-img.shape[2])), constant_values=(bg, bg))
                img_new[i, :, :] = slice_new

            img_data_list.append(img_new)
            sagittal_len.append(img.shape[0])
            transverse_len.append(img.shape[2])
            img_name_list.append(img_name)

        merge_img_data = np.concatenate([i for i in img_data_list], axis=0)

        # merge_img_data1 = merge_img_data.transpose(1,2,0)
        # nft1 = nibabel.Nifti1Image(merge_img_data1, np.eye(4))
        # nibabel.save(nft1, os.path.join('/data/experiments/2D-S/results', 'merge_img_all.nii.gz'))

        return merge_img_data, sagittal_len, transverse_len, img_name_list

    def __getitem__(self, index):
        img = self.merge_img_data[index, :, :]
        sagittal_len = self.sagittal_len
        transverse_len = self.transverse_len
        img_name_list = self.img_name_list

        if self.flag_linear:
            pass
        if np.max(img) > 1:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        # cv2.imwrite(os.path.join('/data/experiments/2D-S/results/img2', str(index)+'.png'), img)

        img = np.expand_dims(img, axis=0)

        # 32bit float
        return torch.FloatTensor(img), sagittal_len, transverse_len, img_name_list

    def __len__(self):
        return self.merge_img_data.shape[0]

class VertebraDataset_2D(Dataset):

    def __init__(self, img_list, subset, prefix='data/', flag_linear=True):
        begin_time = time.time()

        self.img_list = [l.strip() for l in open(img_list).readlines()]
        self.subset = subset
        self.prefix = prefix
        self.flag_linear = flag_linear
        self.randomIdx = self.__randomIdx__()
        self.img_data_list = self.__readAllImgData__()
        self.mask_data_list = self.__readAllMaskData__()
        print('loading %s data %s cost time: %s' % (self.subset, self.mask_data_list.shape, format_time(time.time()-begin_time)))

        # 将训练和验证所用数据保存到单个nii文件
        # nft_img = nibabel.Nifti1Image(self.img_data_list, np.eye(4))
        # nibabel.save(nft_img, os.path.join('/data/experiments/2D-Sagittal', 'all-'+self.subset+'-img.nii.gz'))
        # nft_mask = nibabel.Nifti1Image(self.mask_data_list, np.eye(4))
        # nibabel.save(nft_mask, os.path.join('/data/experiments/2D-Sagittal', 'all-'+self.subset+'-mask.nii.gz'))

    def __randomIdx__(self):
        base = list(np.arange(512))
        randomIdx = []
        for i in range(0, len(self.img_list)):
            randomIdx.append(random.sample(base, 25))
        
        randomIdx = np.array(randomIdx).flatten()
        return randomIdx

    def __readAllImgData__(self):
        img_data_list = []
        for index in range(len(self.img_list)):
            img_name = self.img_list[index].strip()
            img_path = os.path.join(self.prefix, img_name)
            img = nibabel.load(img_path).get_fdata()
            img_new = np.zeros((100, 512, 512)) # if random sampling
            for idx, i in enumerate(range(int(img.shape[0]*0.5-50), int(img.shape[0]*0.5+50))):
                slice = img[i, :, :]
                bg = min(slice.flatten())
                slice_new = np.pad(slice, ((0, 0), (0, 512-img.shape[2])), constant_values=(bg, bg))
                img_new[idx, :, :] = slice_new

            # for idx, i in enumerate(self.randomIdx[index*25 : (index+1)*25]):
            #     slice = img[i, :, :]
            #     bg = min(slice.flatten())
            #     slice_new = np.pad(slice, ((0, 0), (0, 512-img.shape[2])), constant_values=(bg, bg))
            #     img_new[100+idx, :, :] = slice_new

            img_data_list.append(img_new)
        merge_img_data = np.concatenate([i for i in img_data_list], axis=0) # axis=-1, 轴位拼接; axis=0, 矢位拼接
        return merge_img_data

    def __readAllMaskData__(self):
        mask_data_list = []
        for index in range(len(self.img_list)):
            mask_path = os.path.join(self.prefix, 'mask'+self.img_list[index].strip()[4:])
            mask = nibabel.load(mask_path).get_fdata()
            mask_new = np.zeros((100, 512, 512))
            for idx, i in enumerate(range(int(mask.shape[0]*0.5-50), int(mask.shape[0]*0.5)+50)):
                slice = mask[i, :, :]
                bg = min(slice.flatten())
                slice_new = np.pad(slice, ((0, 0), (0, 512-mask.shape[2])), constant_values=(bg, bg))
                mask_new[idx, :, :] = slice_new

            # for idx, i in enumerate(self.randomIdx[index*25 : (index+1)*25]):
            #     slice = mask[i, :, :]
            #     bg = min(slice.flatten())
            #     slice_new = np.pad(slice, ((0, 0), (0, 512-mask.shape[2])), constant_values=(bg, bg))
            #     mask_new[100+idx, :, :] = slice_new

            mask_data_list.append(mask_new)
        merge_mask_data = np.concatenate([i for i in mask_data_list], axis=0)
        return merge_mask_data    

    def __getitem__(self, index):
        img = self.img_data_list[index, :, :]
        mask = self.mask_data_list[index, :, :]

        if self.flag_linear:
            pass
        
        # mask 灰度分布[0, 1]
        mask[mask > 0] = 1
        if np.max(img) > 1:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # 32bit float
        return torch.FloatTensor(img), torch.FloatTensor(mask)

    def __len__(self):
        return self.img_data_list.shape[0]