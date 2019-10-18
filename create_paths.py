import os
import pickle
from utils.readpfm import readPFM
import numpy as np

dir_driving_img='/home/liu/DP_DATA/STEREO/driving_frames_cleanpass'
dir_driving_disp='/home/liu/DP_DATA/STEREO/driving_disparity'

dir_flying_img='/home/liu/DP_DATA/STEREO/frames_cleanpass/TRAIN'
dir_flying_disp='/home/liu/DP_DATA/STEREO/frames_disparity/TRAIN'

dir_monkk_img='/home/liu/DP_DATA/STEREO/monkaa_frames_cleanpass'
dir_monkk_disp='/home/liu/DP_DATA/STEREO/monkaa_disparity'


def add_path(paths, img_path, disp_path):
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if( 'right' in root):
                continue

            img_l = os.path.join(root,file)
            disp_l = img_l.replace(img_path, disp_path)
            disp_l = disp_l.replace('.webp','.npy')
            img_r = img_l.replace('left','right')
            disp_r = disp_l.replace('left','right')
            paths.append({'img_l': img_l, 'disp_l': disp_l,'img_r': img_r, 'disp_r': disp_r})
    return paths


paths=[]

paths = add_path(paths, dir_driving_img, dir_driving_disp)
paths = add_path(paths, dir_flying_img, dir_flying_disp)
paths = add_path(paths, dir_monkk_img, dir_monkk_disp)

paths_80 = []
for path in paths:
    print(path['disp_l'])
    #data, _ = readPFM(path['disp_l'])
    data = np.load(path['disp_l'])
    if(np.max(data)<80.):
        paths_80.append(path)

path_file=open('paths_80.pkl','wb')
pickle.dump(paths,path_file)
path_file.close()
