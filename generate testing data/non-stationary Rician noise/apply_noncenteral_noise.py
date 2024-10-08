import numpy as np
from dipy.io.image import load_nifti, save_nifti

sigma_map,_ = load_nifti('')
idxx = [21,22,23,24,25,26,27,29,30,31,32,33,34]
sigma_map = np.expand_dims(sigma_map,axis=3)
noise_free_path = ''
noisy_free_data, affine = load_nifti(noise_free_path)
noise_real = sigma_map*np.random.standard_normal(noisy_free_data.shape)
noise_complex = sigma_map*np.random.standard_normal(noisy_free_data.shape)
noisy_data = np.sqrt((noisy_free_data + noise_real)**2 + (noise_complex)**2 )
save_path = ''
save_nifti(save_path,np.float32(noisy_data),affine)