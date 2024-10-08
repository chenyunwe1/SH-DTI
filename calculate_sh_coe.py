from dipy.core.gradients import gradient_table
from dipy.core.geometry import cart2sphere
from dipy.core.geometry import cart2sphere, sphere2cart
from scipy.special import sph_harm
from collections import OrderedDict
import numpy as np
from dipy.io.image import load_nifti,save_nifti
import os
import tensorflow as tf

def sh_basis_real(s2_coord, L, even=True):
    """Real spherical harmonic basis of even degree."""
    """
        Arguments:
            s2_coord (numpy array): S2 points coordinates
            L (int): spherical harmonic degree
            even (bool): flag indicating whether to compute only
                spherical harmonics of even degree
        Returns:
            numpy array: spherical harmonic bases (each column is a basis)
    """
    s = 2 if even else 1
    n_sph = np.sum([2 * i + 1 for i in range(0, L + 1, s)])
    Y = np.zeros((s2_coord.shape[0], n_sph), dtype=np.float32)
    n_sph = 0
    for i in range(0, L + 1, s):
        ns, ms = np.zeros(2 * i + 1) + i, np.arange(-i, i + 1)
        Y_n_m = sph_harm(ms, ns, s2_coord[:, 1:2], s2_coord[:, 0:1])
        if i > 0:
            Y_n_m[:, 0:i] = np.sqrt(2) * \
                np.power(-1, np.arange(i, 0, -1)) * np.imag(Y_n_m[:, :i:-1])
            Y_n_m[:, (i + 1):] = np.sqrt(2) * \
                np.power(-1, np.arange(1, i + 1)) * np.real(Y_n_m[:, (i + 1):])
        Y[:, n_sph:n_sph + 2 * i + 1] = Y_n_m
        n_sph += 2 * i + 1
    return Y

def gram_schmidt_sh_inv(s2_coord, L, n_iters=1000, b_type='real', even=True):
    """Inversion of spherical harmonic basis with Gram-Schmidt
       orthonormalization process"""
    """
        Arguments:
            s2_coord (numpy array): S2 points coordinates
            L (int): spherical harmonic degree
            n_iters (float): number of iterations for degree shuffling
        Returns:
            numpy array: inverted spherical harmonic bases
    """
    np.random.seed(1234)
    if b_type == 'real':
        Y = sh_basis_real(s2_coord, L)

    s = 2 if even else 1
    Y_inv_final = np.zeros_like(Y.T)
    for k in range(n_iters):
        order = []
        count_h = 0
        for i in range(0, L + 1, s):
            order_h = count_h + np.arange(0, 2 * i + 1)
            np.random.shuffle(order_h)
            order.extend(list(order_h))
            count_h += 2 * i + 1

        deorder = np.argsort(order)
        Y_inv = np.zeros_like(Y.T)
        for i in range(Y_inv.shape[0]):
            Y_inv[i, :] = Y[:, order[i]]
            for j in range(0, i):
                if np.sum(Y_inv[j, :] ** 2) > 1.e-8:
                    Y_inv[i, :] -= np.sum(Y_inv[i, :] * Y_inv[j, :]) / \
                        (np.sum(Y_inv[j, :] ** 2) + 1.e-8) * Y_inv[j, :]
            Y_inv[i, :] /= np.sqrt(np.sum(Y_inv[i, :] ** 2))
        Y_inv_final += Y_inv[deorder, :]
    Y_inv_final /= n_iters

    n = np.dot(Y[:, 0:1].T, Y[:, 0:1])[0, 0]
    Y_inv_final /= np.sqrt(n)

    return Y_inv_final

# calculate sh basis
def calculate_sh_basis(bvals_name,bvecs_name,layer = 2):
    bvals = np.loadtxt(bvals_name)
    bvecs = np.loadtxt(bvecs_name)
    shell_masks = OrderedDict()
    while np.sum(bvals) > 0:
        bv_max = np.max(bvals)
        sh_mask = (bv_max - bvals) / bv_max == 0
        shell_masks[bv_max] = np.where(sh_mask)[0]
        bvals[shell_masks[bv_max]] = 0
    shell_masks = OrderedDict(sorted(shell_masks.items()))
    gradient_scheme = gradient_table(bvals, bvecs, 100)    
    s2_coords = OrderedDict()
    for k in shell_masks:
        s2_coords[k] = np.zeros((shell_masks[k].shape[0], 3))
        s2_coords_cart = gradient_scheme.bvecs[shell_masks[k], :]

        for i in range(shell_masks[k].shape[0]):
            s2_coords[k][i, :] = cart2sphere(s2_coords_cart[i, 0],
                                            s2_coords_cart[i, 1],
                                            s2_coords_cart[i, 2])
            if s2_coords[k][i, 2] < 0:
                s2_coords[k][i, 2] = 2 * np.pi + s2_coords[k][i, 2]
    coords = np.zeros((shell_masks[k].shape[0], 3, 1),dtype=np.float32)
    coords[:, :,0] = s2_coords[k]
    points_sph = coords[:,1:,0]
    points_cart = np.zeros((points_sph.shape[0], 3))
    for i in range(points_sph.shape[0]):
        theta, phi = points_sph[i, :]
        x, y, z = sphere2cart(1, theta, phi)
        points_cart[i, 0] = x
        points_cart[i, 1] = y
        points_cart[i, 2] = z
    coords_cart = {}
    coords_cart[0] = points_cart
    for i in range(coords_cart[0].shape[0]):
        n = np.sqrt(np.sum(coords_cart[0][i, :] ** 2))
        coords_cart[0][i, :] /= n

    points_cart = coords_cart[0]
    points_s2 = np.zeros((points_cart.shape[0], 2))
    for i in range(points_cart.shape[0]):
        x, y, z = points_cart[i, :]
        r, theta, phi = cart2sphere(x, y, z)
        points_s2[i, 0] = theta
        points_s2[i, 1] = phi

    coords_sph = {}
    coords_sph[0] = points_s2
    print(shell_masks)
    n_sph = np.sum([2 * l + 1 for l in range(0, layer + 1, 2)])
    Y = np.zeros((1, n_sph, coords_sph[0].shape[0]))
    Y[0,:,:] = gram_schmidt_sh_inv(coords_sph[0],L=layer,b_type='real',n_iters=1000)    
    return Y

def calculate_sh_maps(data,mask,Y_inv,bval=1000):
    # b0 normalization
    b0_nor_noisy = (data[:,:,:,1:data.shape[3]]/(data[:,:,:,0:1]+1e-10))**(1000/bval)
    b0_nor_noisy[b0_nor_noisy>1]=1
    b0_nor_noisy[b0_nor_noisy<0]=0                            
    b0_nor_noisy = np.expand_dims(b0_nor_noisy,axis=4)
    b0_nor_noisy = tf.convert_to_tensor(b0_nor_noisy, dtype=tf.float32)
    sh_coe_map = tf.squeeze(tf.matmul(Y_inv,b0_nor_noisy))
    mask = np.expand_dims(mask,axis=3)  
    sh_coe_map = mask*np.array(sh_coe_map)
    sh_coe_map = sh_coe_map.astype(np.float32)    
    
    return sh_coe_map

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    bvals_name = ''
    bvecs_name = ''
    mask_name = ''
    data_name = ''
    data, affine = load_nifti(data_name)
    mask, _ = load_nifti(mask_name)
    # calculate Spherical harmonics basis
    Y_inv = calculate_sh_basis(bvals_name,bvecs_name,layer = 2)
    # calculate Spherical harmonics coefficients
    sh_coe_map = calculate_sh_maps(data,mask,Y_inv,bval=1000)
    save_name = ''
    save_nifti(save_name,sh_coe_map,affine)