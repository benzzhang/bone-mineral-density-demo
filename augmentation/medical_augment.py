'''
Author: Pengbo
LastEditTime: 2022-05-21 18:59:00
Description: augmentation for landmark detection of medical image
'''
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import transform as sktrans
import albumentations as A

def rotate(angle):
    '''
        angle: °
    '''
    def func(img):
        ''' img: ndarray, channel x imgsize
        '''
        ret = []
        for i in range(img.shape[0]):
            ret.append(sktrans.rotate(img[i], angle))
        return np.array(ret)
    return func


def translate(offsets):
    ''' translation
        offsets: n-item list-like, for each dim
    '''
    offsets = tuple(offsets)
    new_sls = tuple(slice(i, None) for i in offsets)

    def func(img):
        ''' img: ndarray, channel x imgsize
        '''
        ret = []
        size = img.shape[1:]
        old_sls = tuple(slice(0, j-i) for i, j in zip(offsets, size))

        for old in img:
            new = np.zeros(size)
            new[new_sls] = old[old_sls]
            ret.append(new)
        return np.array(ret)
    return func


def flip(axis=1):
    '''
    axis=0: flip all
       else flip axis
    '''
    f_sls = slice(None, None, -1)
    sls = slice(None, None)

    def func(img):
        dim = img.ndim
        cur_axis = axis % dim
        if cur_axis == 0:
            all_sls = tuple([f_sls])*dim
        else:
            all_sls = tuple(
                            f_sls if i == cur_axis else sls for i in range(dim))
            return img[all_sls]
    return func

def gaussian_blur(image):
    return gaussian_filter(image, sigma=1)

def gaussian_noise(image):
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = 50 * np.random.normal(mean, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    return image + gauss

def elastic_transform(image, mask, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

       Modified from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    distored_mask = map_coordinates(mask, indices, order=1, mode='reflect')

    return distored_image.reshape(image.shape), distored_mask.reshape(mask.shape)

def elastic_transform_from_2D(image, mask, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    print('image:',image.shape)
    for z in range(image.shape[2]):
        image_slice = image[:, :, z]
        mask_slice = mask[:, :, z]
        # 参数 sigma， alpha需要调整，微小形变
        transformed = A.ElasticTransform(alpha, sigma)(image=image_slice, mask=mask_slice)
        image_elastic = np.reshape(transformed['image'],(image.shape[0],image.shape[1],1))
        mask_elastic = np.reshape(transformed['mask'],(image.shape[0],image.shape[1],1))
        if z==0:
            image_temp = image_elastic
            mask_temp = mask_elastic
        else:
            image_temp = np.concatenate([image_temp, image_elastic], axis=2)
            mask_temp = np.concatenate([mask_temp, mask_elastic], axis=2)
    print('temp:',image_temp.shape)
    return image_temp, mask_temp