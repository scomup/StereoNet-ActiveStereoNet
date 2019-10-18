import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.color import rgb2gray

from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix


def get_L(img,patch_size):
    H, W = img.shape
    all_pixel = patch_size*patch_size
    radius = patch_size
    coord_cells = np.stack(np.meshgrid(np.arange(0,patch_size), np.arange(0,patch_size),indexing='ij'), axis=2).astype(np.float32)
    pixel_distances = np.linalg.norm(coord_cells.reshape(1,all_pixel,2)-coord_cells.reshape(all_pixel,1,2), axis=-1)
    patch_num = (H//patch_size)*(W//patch_size)

    L_list = []
    for j in range(patch_num):
        x = get_img_patch(img,patch_size,j).reshape(-1)
        A = x.reshape(all_pixel,1)
        B = x.reshape(1,all_pixel)
        intensity_distances = (x.reshape(all_pixel,1) - x.reshape(1,all_pixel))**2
        weight = intensity_distances *255. + pixel_distances * .2
        weight_mask = weight < 16.
        weight = np.exp(-weight) * weight_mask
        L = np.sum(weight,axis=1)*np.eye(all_pixel) - weight
        L_list.append(L)
    return L_list




    

def get_img_patch(img, patch_size,  patch_id):
    _, W = img.shape
    pW = W//patch_size
    w = patch_id%pW * patch_size
    h = patch_id//pW * patch_size
    return img[h:h+patch_size,w:w+patch_size]

def put_img_patch(img, patch_size, patch_id, patch):
    _, W = img.shape
    pW = W//patch_size
    w = patch_id%pW * patch_size
    h = patch_id//pW * patch_size
    img[h:h+patch_size,w:w+patch_size] = patch


img = io.imread('/home/liu/DP_DATA/COCO/train2014/COCO_train2014_000000000036.jpg')
img = rgb2gray(img)

patch_size = 20
H=200
W=200
patch_num = (H//patch_size)*(W//patch_size)
use_sparse = False
assert 0 == H%patch_size, 'error H'
assert 0 == W%patch_size, 'error W'

img = transform.resize(img, (H, W))
img_0 = np.copy(img)
#L = get_L(img, patch_size,7.5,use_sparse)

#(y-x)^2 + l * xT * L * x 

#E = sparse.eye(patch_size*patch_size).tocsc()
E=np.eye(patch_size*patch_size)

for i in range(10):
    cost = 0
    L_list = get_L(img, patch_size)

    for j in range(patch_num):
        x = get_img_patch(img,patch_size,j)
        x = np.matrix(x).reshape(-1).transpose()
        L = L_list[j]
        #print("cost:%f"%cost)
        M = E + 0.8*L
        M_inv = np.linalg.inv(M)
        y = M_inv * x

        put_img_patch(img,patch_size,j,y.reshape(patch_size,patch_size))
        cost += np.linalg.norm(x-y)**2 
        #print("patch:%d"%j)
    print("cost:%f"%cost)

    plt.imshow(np.concatenate([img_0,img]),cmap='Greys_r',vmin=0,vmax=1)
    plt.savefig('figure%d.png'%i) 
    plt.show()


plt.imshow(L)
plt.colorbar()
plt.show()


