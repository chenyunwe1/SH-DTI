from dipy.core.gradients import gradient_table
from dipy.core.geometry import cart2sphere
from dipy.core.geometry import cart2sphere, sphere2cart
from scipy.special import sph_harm
from collections import OrderedDict
import numpy as np
from dipy.io.image import load_nifti
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

    return Y_inv_final,Y

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

# select patch and add noise
def mk_train_dataset(train_data_path,nf_data_name,mask_name,noisy_data_save_path,gt_save_path):
    idx = 0
    for file_name in os.listdir(train_data_path):
        print(file_name)
        nf_data_path = train_data_path + '/' + file_name + '/' + nf_data_name
        mask_path = train_data_path + '/' + file_name + '/' + mask_name
        nf_data,_ = load_nifti(nf_data_path)
        mask,_ = load_nifti(mask_path)
        for i in range(7):
            noise_level = np.linspace(0.01,0.03,35)
            random_choice = np.random.choice(35, 35, replace=False)
            noise_level = noise_level[random_choice]
            idx_0 = 0
            for j in range(7):
                for k in range(5):
                    print(idx)
                    print('add noise')
                    nf_data_patch = nf_data[16*i:16*i+32, 16*j:16*j+32, 16*k:16*k+32, :] 
                    mask_patch = mask[16*i:16*i+32, 16*j:16*j+32, 16*k:16*k+32]
                    noise_level_patch = noise_level[idx_0]
                    idx_0 = idx_0 + 1
                    noise_r_patch = noise_level_patch*np.random.normal(0,1,size=nf_data_patch.shape)
                    noise_c_patch = noise_level_patch*np.random.normal(0,1,size=nf_data_patch.shape)
                    noisy_data_patch = np.sqrt((nf_data_patch + noise_r_patch)**2 + noise_c_patch**2)
                    gt_patch = np.zeros([32,32,32,16])
                    b0_nor_gt = nf_data_patch[:,:,:,1:16] /(nf_data_patch[:,:,:,0:1]+1e-10)
                    b0_nor_gt[b0_nor_gt<0]=0
                    b0_nor_gt[b0_nor_gt>1]=1
                    gt_patch[:,:,:,0:15] = b0_nor_gt
                    gt_patch[:,:,:,15] = mask_patch
                    save_path_dwi = os.path.join(noisy_data_save_path, 'dwi_nd_%04d.npy' % idx)
                    save_path_gt = os.path.join(gt_save_path, 'gt_%04d.npy' % idx)
                    idx = idx + 1
                    noisy_data_patch = noisy_data_patch.astype(np.float32)
                    gt_patch = gt_patch.astype(np.float32)
                    np.save(save_path_dwi,noisy_data_patch)
                    np.save(save_path_gt,gt_patch)
    flag = 1
    return flag


# spherical harmonics representation
def spherical_harmonics_representation(Y_inv,noisy_data_save_path,gt_save_path,sh_coe_save_path):
    Y_inv = tf.convert_to_tensor(Y_inv, dtype=tf.float32)
    for i in range(3920):
        dwi_nd_name = os.path.join(noisy_data_save_path, 'dwi_nd_%04d.npy' % i)
        gt_name = os.path.join(gt_save_path, 'gt_%04d.npy' % i)
        gt_nf = np.load(gt_name)
        dwi_nd = np.load(dwi_nd_name)
        mask = gt_nf[:,:,:,15]
        b0_nor_noisy = dwi_nd[:,:,:,1:16]/(dwi_nd[:,:,:,0:1]+1e-10)
        b0_nor_noisy[b0_nor_noisy>1]=1
        b0_nor_noisy[b0_nor_noisy<0]=0                            
        b0_nor_noisy = np.expand_dims(b0_nor_noisy,axis=4)
        b0_nor_noisy = tf.convert_to_tensor(b0_nor_noisy, dtype=tf.float32)
        sh_coe_map = tf.squeeze(tf.matmul(Y_inv,b0_nor_noisy))
        mask = np.expand_dims(mask,axis=3)  
        sh_coe_map = mask*np.array(sh_coe_map)
        sh_coe_map = sh_coe_map.astype(np.float32)
        save_path_sph = os.path.join(sh_coe_save_path, 'sph_%04d.npy' % i)
        np.save(save_path_sph,sh_coe_map)
    flag = 1
    return flag
    
# save train dataset as tfrecoreds format
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write2TFRecord(img_name,filename,sh_coe_save_path,gt_save_path):
    filename = filename+'.tfrecords'
    writer = tf.io.TFRecordWriter(filename) # writer taht will store data to disk
    for i in range(len(img_name)):
        
        img_path = os.path.join(sh_coe_save_path, 'sph_%04d.npy' % img_name[i])
        patches_imgs = np.load(img_path)
        Nx,Ny,Nz,Nc = patches_imgs.shape
        gt_path = os.path.join(gt_save_path, 'gt_%04d.npy' % img_name[i])
        patches_gts = np.load(gt_path)
        Nx,Ny,Nz,Nc2 = patches_gts.shape
        feature = {
            'patches_imgs': _bytes_feature(tf.io.serialize_tensor(patches_imgs)),
            'patches_gts': _bytes_feature(tf.io.serialize_tensor(patches_gts)),
            'Nx': _int64_feature(Nx),
            'Ny': _int64_feature(Ny),
            'Nz': _int64_feature(Nz),
            'Nc': _int64_feature(Nc),
            'Nc2': _int64_feature(Nc2),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        example = example.SerializeToString()
        writer.write(example)
    writer.close()
    print('Wrote '+str(len(img_name))+' elemets to TFRecord')


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    train_data_path = ''
    nf_data_name = ''
    bvals_name = ''
    bvecs_name = ''
    mask_name = ''
    noisy_data_save_path = ''
    gt_save_path = ''
    sh_coe_save_path = ''
    tfrecord_path = ''
    tfrecord_filename=os.path.join(tfrecord_path,'train')
    # calculate Spherical harmonics basis
    Y_inv = calculate_sh_basis(bvals_name,bvecs_name,layer = 2)
    # select patch and add noise
    a = mk_train_dataset(train_data_path,nf_data_name,mask_name,noisy_data_save_path,gt_save_path)
    print(a)
    # spherical harmonics representation
    b = spherical_harmonics_representation(Y_inv,noisy_data_save_path,gt_save_path,sh_coe_save_path)
    print(b)
    # save train dataset as tfrecoreds format
    idx = np.random.choice(3920, 3920, replace=False)
    write2TFRecord(idx,tfrecord_filename,sh_coe_save_path,gt_save_path)