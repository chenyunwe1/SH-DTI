# Spherical harmonics-based deep learning achieves generalized and accurate diffusion tensor imaging

![The overview of SH-DTI](https://github.com/chenyunwe1/SH-DTI/blob/main/Overview.png)

**The overview of SH-DTI**. (a), The training process of SH-DTI. SH-DTI learns the mapping from low-quality spherical harmonics coefficient maps to high-quality diffusion tensor. (b), Experimental design. Extensive experiments were conducted on both simulated and in-vivo datasets, covering various DTI application scenarios.

## mk_nf_data.m
The code used for generating the noise-free dMRI data.

**Output**
- *dwi.nii.gz* - Noise-free dMRI data.

## make_train_dataset.py
The code used for preparing the training dataset for the network in SH-DTI.

**Output**
- *training_dataset.tfrecords* - Input and ground-truth data prepared for network.

## make_valid_dataset.py
The code used for preparing the validation  dataset for the network in SH-DTI.

**Output**
- *sph_xxxx.npy* - Input data prepared for network.
- *gt_xxxx.npy* - Ground-truth data prepared for network.

## train.py
The code for training the network in SH-DTI using datasets prepared using the make_train_dataset.py and make_valid_dataset.py scripts.

**Output**
- *model_xxx.h5* - The model saved every five training iterations.

## Trained model
Due to the 25MB file upload limit, the trained model cannot be uploaded. If you need, please download it at https://drive.google.com/file/d/1103ntSQ3KYY8AHoJO_Bbak2-tcA9dncY/view?usp=drive_link.

## test.py
The code for tesing the trained SH-DTI model.

**Output**
- *tensor.nii.gz* - The estimated diffusion tensor (Dxx, Dyy, Dzz, Dxy, Dxz, and Dyz) with a shape of [x, y, z, 6]. 
