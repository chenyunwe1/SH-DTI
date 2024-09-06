import tensorflow as tf
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SH-DTI', type=str, help='model name')
parser.add_argument('--order', default='2-order', type=str, help='data name')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--epoch', default=150, type=int, help='number of train epoches')
parser.add_argument('--lr', default=2.5e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
args = parser.parse_args() 

# model path
path = ''
model_dir = os.path.join(args.data_name, path)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

save_dir = os.path.join(path, args.model,args.order)  # change!!
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
# network
def UNet(image_channels):
    inpt = tf.keras.layers.Input(shape=(None, None, None, image_channels))
    conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inpt)
    conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(conv1) 

    conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(conv2)  

    conv3 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))(conv3)  


    convbase = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
    convbase = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(convbase)  
    conc5 = tf.keras.layers.Concatenate()([
        tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same')(
            tf.keras.layers.UpSampling3D(size=(2, 2, 2))(convbase)), conv3])

    conv6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc5)
    conv6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
    conc7 = tf.keras.layers.Concatenate()([
        tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same')(
            tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv6)), conv2]) 

    conv7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc7)
    conv7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
    conc8 = tf.keras.layers.Concatenate()([
        tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(
            tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv7)), conv1])  
    conv8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc8)
    conv8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)
    convone = tf.keras.layers.Conv3D(6, (1, 1, 1))(conv8)
    model = tf.keras.Model(inputs=inpt, outputs=convone)
    return model

def U_DTI(image_channels=6):
    model_mapping = UNet(image_channels=image_channels)                 
    inpts = tf.keras.Input(shape=(None, None, None, image_channels))     
    mapping_pred = model_mapping(inpts)                                  
    model = tf.keras.Model(inputs=inpts, outputs=[mapping_pred])         
    return model

def findLastCheckpoint(save_dir): 
    import glob, re  
    file_list = glob.glob(os.path.join(save_dir, 'model_*.h5'))  
    if file_list:  
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).h5.*", file_) 
            # print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

# learning rate strategy
def lr_schedule(epoch):  
    initial_lr = args.lr
    if epoch <= 50:
        lr = initial_lr
    elif epoch <= 100:
        lr = initial_lr * 0.5    
    elif epoch <= 150:
        lr = initial_lr * 0.3   
    else:
        lr = initial_lr * 0.1 
    return lr

# loss function
def loss_model_exp_image(y_true, y_pred): 
    mask = y_true[:,:,:,:,15]
    y_true = y_true[:,:,:,:,0:15]
    mask = tf.expand_dims(mask, axis=4)
    mask = tf.concat([mask,mask,mask,mask,mask,mask,mask,mask,mask,mask,mask,mask,mask,mask,mask],axis=4)
    idx = tf.where(mask>0)
    # load diffusion gradient scheme
    bvecs_filepath = r''
    bvecs_np = np.loadtxt(bvecs_filepath).astype('float32')
    bvecs = tf.convert_to_tensor(bvecs_np)

    d1 = tf.expand_dims(y_pred[:, :, :, :, 0],axis=4)
    d2 = tf.expand_dims(y_pred[:, :, :, :, 1],axis=4)
    d3 = tf.expand_dims(y_pred[:, :, :, :, 2],axis=4)
    d4 = tf.expand_dims(y_pred[:, :, :, :, 3],axis=4)
    d5 = tf.expand_dims(y_pred[:, :, :, :, 4],axis=4)
    d6 = tf.expand_dims(y_pred[:, :, :, :, 5],axis=4)

    d_9dir = d1  
    d_9dir = tf.concat([d_9dir, d4], -1)  
    d_9dir = tf.concat([d_9dir, d5], -1)
    d_9dir = tf.concat([d_9dir, d4], -1)
    d_9dir = tf.concat([d_9dir, d2], -1)
    d_9dir = tf.concat([d_9dir, d6], -1)
    d_9dir = tf.concat([d_9dir, d5], -1)
    d_9dir = tf.concat([d_9dir, d6], -1)
    d_9dir = tf.concat([d_9dir, d3], -1)  
    d = tf.reshape(d_9dir, [4,32,32,32,3,3])
    d = tf.cast(d, dtype=tf.float32)

    for dir in range(0, 15):
        precon = tf.squeeze(tf.exp(0.001 * -1000.0 * tf.matmul(tf.matmul(tf.reshape(bvecs[dir,:],[1,3]),d),tf.reshape(bvecs[dir,:],[3,1]))))
        precon = tf.expand_dims(precon,axis=4)
        if dir == 0:
            precon1 = precon
        else:
            precon1 = tf.concat([precon1, precon], axis=4)
    recon = precon1
    recon_idx = tf.gather_nd(recon, idx)
    y_true_idx = tf.gather_nd(y_true, idx)
    loss = tf.keras.losses.mean_squared_error(recon_idx, y_true_idx)
    return loss

def parse_all(example):
    feature_discription={
        'patches_imgs': tf.io.FixedLenFeature([], tf.string),
        'patches_gts': tf.io.FixedLenFeature([], tf.string),
        'Nx': tf.io.FixedLenFeature([], tf.int64),
        'Ny': tf.io.FixedLenFeature([], tf.int64),
        'Nz': tf.io.FixedLenFeature([], tf.int64),
        'Nc': tf.io.FixedLenFeature([], tf.int64),        
        'Nc2': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example,feature_discription)

    Nx = parsed_example['Nx']
    Ny = parsed_example['Ny']
    Nz = parsed_example['Nz']
    Nc = parsed_example['Nc']
    Nc2 = parsed_example['Nc2']
    
    patches_imgs    = tf.io.parse_tensor(parsed_example['patches_imgs'],out_type=tf.float32)
    patches_gts    = tf.io.parse_tensor(parsed_example['patches_gts'],out_type=tf.float32)
    patches_imgs    = tf.reshape(patches_imgs,shape=[Nx,Ny,Nz,Nc])
    patches_gts    = tf.reshape(patches_gts,shape=[Nx,Ny,Nz,Nc2])
    return (patches_imgs,patches_gts)

def get_valid_data_patch():
    valid_data = np.zeros([980,32,32,32,6],dtype='float32')
    valid_gt = np.zeros([980,32,32,32,16],dtype='float32')  
    for i in range(980):
        valid_input_data_path = ''
        input_data_path = os.path.join(valid_input_data_path, 'sph_%04d.npy' % i)
        print(input_data_path)
        input_data = np.load(input_data_path).astype(np.float32)
        
        valid_ground_truth_path = ''
        ground_truth_path = os.path.join(valid_ground_truth_path, 'gt_%04d.npy' % i)
        ground_truth = np.load(ground_truth_path).astype(np.float32)
        valid_data[i,:,:,:,:] = input_data
        valid_gt[i,:,:,:,:] = ground_truth    
    valid_random_slice = np.random.choice(980, 980, replace=False)
    return valid_data[valid_random_slice,:,:,:,:],valid_gt[valid_random_slice,:,:,:,:]

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    
    # load valid data
    valid_data, valid_gt = get_valid_data_patch()
    valid_y = valid_gt
    valid_x = valid_data
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))    
    valid_ds = valid_ds.batch(args.batch_size, drop_remainder=True)
    
    # load training data
    path_train = ''
    train_filenames = tf.io.gfile.glob(os.path.join(path_train,'train.tfrecords'))
    dataset_train = tf.data.TFRecordDataset(train_filenames).map(parse_all)
    
    # network
    model = U_DTI()
    model.summary()
    
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = tf.keras.models.load_model(os.path.join(save_dir, 'model_%03d.h5' % initial_epoch), compile=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(),   
                  loss=[loss_model_exp_image],
                  loss_weights=[1.0])


    checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
                                                      verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'log.csv'), append=True, separator=',')     
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)                      
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=save_dir, histogram_freq=1)         
    
    # train
    history = model.fit(x = dataset_train.repeat(args.epoch).batch(args.batch_size,drop_remainder=True), steps_per_epoch  = 980,
                        epochs=args.epoch, 
                        verbose=1, 
                        initial_epoch=initial_epoch,  
                        callbacks=[checkpointer, csv_logger, lr_scheduler, tensorboard],
                        validation_data = valid_ds)