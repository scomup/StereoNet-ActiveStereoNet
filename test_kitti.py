import os
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2 
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models.StereoNet8Xmulti import StereoNet
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True

dir_kitti='/home/liu/DP_DATA/STEREO/KITTI/testing/image_2'


paths=[]
for root, dirs, files in os.walk(dir_kitti):
    for file in files:
        paths.append(os.path.join(root,file))

net = StereoNet(3,3,192)
#net=net.cuda()
net = torch.nn.DataParallel(net).cuda()

checkpoint = torch.load('/home/liu/workspace/StereoNet-ActiveStereoNet/results/8Xmulti/checkpoint.pth')
net.load_state_dict(checkpoint['state_dict'])



mean = torch.tensor([0., 0., 0.], dtype=torch.float32)
std = torch.tensor([1., 1., 1.], dtype=torch.float32)

totensor = transforms.ToTensor()
normalize = transforms.Normalize(mean.tolist(), std.tolist())
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


fig, ax = plt.subplots(2, 2, figsize=(16, 8))
import time

for path_left in paths:
    path_right = path_left.replace('image_2', 'image_3')
    imageL = cv2.imread(path_left)
    imageR = cv2.imread(path_right)

    imageL = cv2.resize(imageL, dsize=None, fx=1.5, fy=1.5)
    imageR = cv2.resize(imageR, dsize=None, fx=1.5, fy=1.5)

    #new_size = (w, h)
    #imageL = cv2.resize(imageL, new_size, interpolation=cv2.INTER_NEAREST)
    #imageR = cv2.resize(imageR, new_size, interpolation=cv2.INTER_NEAREST)
    imageL = normalize(totensor(imageL))[None,:].cuda()
    imageR = normalize(totensor(imageR))[None,:].cuda()
    start = time.time()
    with torch.no_grad():
        result = net(imageL, imageR)
    end = time.time()
    print(end - start)
    if(True):
        imL_ = unnormalize(imageL[0]).permute(1,2,0).cpu().detach().numpy()
        imR_ = unnormalize(imageR[0]).permute(1,2,0).cpu().detach().numpy()
        disp_NET_ = result[3].cpu().detach().numpy()[0]
        
        #plt.subplot(2,2,1)
        ax[0,0].imshow( imL_[...,::-1])
        #plt.subplot(2,2,2)
        ax[0,1].imshow(imR_[...,::-1])
        #plt.subplot(2,2,4)
        ax[1,0].imshow(disp_NET_*2,cmap='rainbow',vmin=0, vmax=192)
        #plt.colorbar()
        plt.pause(1)
    

    
