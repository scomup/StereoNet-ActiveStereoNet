import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.color import rgb2gray

from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix


def get_L(img,radius):
    H, W = img.shape
    max_r = int(np.ceil(radius))
    coord_cells = np.stack(np.meshgrid(np.arange(-max_r,max_r+1), np.arange(-max_r,max_r+1),indexing='ij'), axis=2).astype(np.float32)
    pixel_distances = np.linalg.norm(coord_cells, axis=-1)
    pixel_distances[max_r,max_r] += radius #remove self.
    pixel_in_radius_i,pixel_in_radius_j  = np.where(pixel_distances < radius)
    weight_table = np.exp(-pixel_distances)[pixel_in_radius_i,pixel_in_radius_j].reshape(-1)
    pixel_in_radius_i = pixel_in_radius_i - max_r
    pixel_in_radius_j = pixel_in_radius_j - max_r
    #G = np.zeros([H*W,H*W])
    L=lil_matrix((H*W,H*W))

    for col in range(L.shape[1]):
        i = pixel_in_radius_i + int(col / W)
        j = pixel_in_radius_j + int(col % W)
        mask = (i>=0) * (i<H) * (j>=0) * (j<W)
        mask = (i>=0) * (i<H) * (j>=0) * (j<W)
        pixels = (i*W+j)[mask]
        weights = weight_table[mask]
        #G[col, pixels] = weights
        L[col, pixels] = -weights
        L[col, col] = np.sum(weights)

    return L.tocsc()


img = io.imread('/home/liu/DP_DATA/COCO/train2014/COCO_train2014_000000000036.jpg')
img = rgb2gray(img)
img = transform.resize(img, (120,160))

L = get_L(img, 7.5)

#(y-x)^2 + l * xT * L * x 
H, W = img.shape
E = sparse.eye(H*W).tocsc()
x = np.matrix(img.reshape(H*W,1))

for i in range(10):
    cost = x.transpose()*L *x
    print("cost:%f"%cost)
    M = E + 0.2*L
    M_inv = sparse.linalg.inv(M)
    dx = M_inv * x
    #sM = sparse.csr_matrix(M)
    #dx = np.linalg.inv((E + 0.2*x.transpose()*L)) * x
    x = x - dx * 0.1
    #img = x.reshape(H,W)
    plt.imshow(M)
    plt.show()


plt.imshow(L.todense())
plt.colorbar()
plt.show()


