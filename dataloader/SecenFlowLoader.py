import torch.utils.data as data
import random
from PIL import Image
from . import preprocess
# import preprocess
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]

from torchvision.transforms.functional import adjust_brightness, adjust_contrast
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2 as cv


def motion_blur(image, max_kernel_size=20):

    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    image = cv.filter2D(image, -1, kernel)
    return image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


# def disparity_loader(path):
#     path_prefix = path.split('.')[0]
#     # print(path_prefix)
#     path1 = path_prefix + '_exception_assign_minus_1.npy'
#     path2 = path_prefix + '.npy'
#     path3 = path_prefix + '.pfm'
#     import os.path as ospath
#     if ospath.exists(path1):
#         return np.load(path1)
#     else:
#         if ospath.exists(path2):
#             data = np.load(path2)
#         else:
#             # from readpfm import readPFMreadPFM
#             from readpfm import readPFM
#             data, _ = readPFM(path3)
#             np.save(path2, data)
#         for i in range(data.shape[0]):
#             for j in range(data.shape[1]):
#                 if j - data[i][j] < 0:
#                     data[i][j] = -1
#         np.save(path1, data)
#         return data


def disparity_loader(path):
    path_prefix = path.split('.')[0]
    # print(path_prefix)
    path1 = path_prefix + '_exception_assign_minus_1.npy'
    path2 = path_prefix + '.npy'
    path3 = path_prefix + '.pfm'
    import os.path as ospath
    if ospath.exists(path1):
        return np.load(path1)
    else:

        # from readpfm import readPFMreadPFM
        from readpfm import readPFM
        data, _ = readPFM(path3)
        np.save(path2, data)
        #for i in range(data.shape[0]):
        #    for j in range(data.shape[1]):
        #        if j - data[i][j] < 0:
        #            data[i][j] = -1
        np.save(path1, data)
        return data

class myImageFloder(data.Dataset):
    def __init__(self,
                 left,
                 right,
                 left_disparity,
                 training,
                 normalize,
                 loader=default_loader,
                 dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.normalize = normalize

    def __getitem__(self, index):
        
        left = self.left[index]
        
        right = self.right[index]
        disp_L = self.disp_L[index]
        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)
        dataL = np.clip(dataL, 0, 192)   
        stddev = np.random.uniform(5,10)

        factor = 1. + random.uniform(-0.5, 0.5)
        left_img = adjust_brightness(left_img, factor)
        right_img = adjust_brightness(right_img, factor)


        factor = 1. + random.uniform(-0.5, 0.5)
        left_img = adjust_contrast(left_img, factor)
        right_img = adjust_contrast(right_img, factor)


        left_img = np.asarray(left_img)
        right_img = np.asarray(right_img)
        

        noise = np.random.normal(0, stddev, left_img.shape)
        left_img = np.clip(left_img + noise, 0, 255)
        noise = np.random.normal(0, stddev, left_img.shape)
        right_img = np.clip(right_img + noise, 0, 255)

        left_img = motion_blur(left_img)
        right_img = motion_blur(right_img)


        left_img = Image.fromarray(np.uint8(left_img))
        right_img = Image.fromarray(np.uint8(right_img))


        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        processed = preprocess.get_transform(
            augment=False, normalize=self.normalize)
        left_img = processed(left_img)
        right_img = processed(right_img)


        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
if __name__ == '__main__':
    path = '/media/lxy/sdd1/stereo_coderesource/dataset_nie/SceneFlowData/frames_cleanpass/flyingthings3d_disparity/TRAIN/A/0024/left/0011.pfm'
    res = disparity_loader(path)
    print(res.shape)
