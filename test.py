import os
import numpy as np
import tensorflow as tf
from dipy.io.image import load_nifti, save_nifti

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
model_name = ''
model = tf.keras.models.load_model(model_name, compile=False)
data_name = ''
input,affine = load_nifti(data_name)
input = np.expand_dims(input,axis=0)
output = np.squeeze(model.predict(input)) 
save_name = ''
save_nifti(save_name,output,affine)
