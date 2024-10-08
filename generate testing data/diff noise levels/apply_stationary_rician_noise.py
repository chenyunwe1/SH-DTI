import numpy as np
from dipy.io.image import load_nifti,save_nifti
# noise level  sigma can be set to 0.01, 0.02, and 0.03.
sigma = 0.02
# noise-free data
dwi_path = ''
dwi_nf,affine = load_nifti(dwi_path)
# apply rician noise  
noise_level_r = sigma*np.random.standard_normal(dwi_nf.shape)
noise_level_c = sigma*np.random.standard_normal(dwi_nf.shape)
dwi_noise = np.sqrt((dwi_nf + noise_level_r)**2 + noise_level_c**2)
# save noisy data
save_path = ''
save_nifti(save_path,np.float32(dwi_noise),affine)
