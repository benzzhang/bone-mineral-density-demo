import os
import nibabel
import cv2
import numpy as np
import imageio
from utils import progress_bar

def nii2png(src_path, dst_path, size=(512, 512)):
    print('==> Transforming .nii to .png')
    files = os.listdir(src_path)
    image_list = [i for i in files if 'seg' not in i]

    for idx, file in enumerate(image_list):
        image_path = os.path.join(src_path, file)
        mask_path = os.path.join(src_path, file.split('.')[0]+'_seg.nii.gz')
        imgae = nibabel.load(image_path).get_fdata()
        mask = nibabel.load(mask_path).get_fdata()
        quatern_b = round(abs(nibabel.load(image_path).header['quatern_b']), 1)
        if quatern_b == 0.5:
            for i in range(imgae.shape[1]):
                slice = imgae[:, i, :]
                slice = cv2.resize(slice, size)
                slice = np.expand_dims(slice, axis=-1)
                cv2.imwrite(os.path.join(dst_path, 'image/'+file.split('.')[0]+'_'+str(i)+'.png'), slice)

                slice = mask[:, i, :]
                slice = cv2.resize(slice, size)
                slice = np.expand_dims(slice, axis=-1)
                # imageio - 输出按比例scale到255， cv2 - 输出真实值
                imageio.imwrite(os.path.join(dst_path, 'mask/'+file.split('.')[0]+'_'+str(i)+'.png'), slice)
                # cv2.imwrite(os.path.join(dst_path, 'mask/'+file.split('.')[0]+'_'+str(i)+'.png'), slice)
        else:
            for i in range(imgae.shape[2]):
                slice = imgae[:, :, i]
                slice = cv2.resize(slice, size)
                slice = np.expand_dims(slice, axis=-1)
                cv2.imwrite(os.path.join(dst_path, 'image/'+file.split('.')[0]+'_'+str(i)+'.png'), slice)
                
                slice = mask[:, :, i]
                slice = cv2.resize(slice, size)
                slice = np.expand_dims(slice, axis=-1)
                imageio.imwrite(os.path.join(dst_path, 'mask/'+file.split('.')[0]+'_'+str(i)+'.png'), slice)
                # cv2.imwrite(os.path.join(dst_path, 'mask/'+file.split('.')[0]+'_'+str(i)+'.png'), slice)
        
        progress_bar(idx, len(image_list))

if __name__ == '__main__':
    src_path = '/data/VerSe_dataset_128'
    dst_path = '/data/VerSe_pngs'
    nii2png(src_path, dst_path)
