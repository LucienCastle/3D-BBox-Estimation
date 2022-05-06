import numpy as np
import os
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import adam_v2
from loss_func import orientation_loss
from preprocess_data import load_and_process_annotation_data,train_data_gen
from model import bbox_3D_net

# gpu if available
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# instantiate model
model = bbox_3D_net((224,224,3),bin_num=6,weights='imagenet')

# to save model architecture
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_arch.png', show_shapes=True)

model.summary()

# optimizer
minimizer = adam_v2.Adam(lr=1e-5)

# define early stopping condition and checkpoint to save model weights after each epoch
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='min', verbose=1)
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
tensorboard = TensorBoard(log_dir='../logs/', histogram_freq=0, write_graph=True, write_images=False)

# compile model
model.compile(optimizer=minimizer,#minimizer,
              loss={'dimension': 'mean_squared_error', 'orientation': orientation_loss, 'confidence': 'categorical_crossentropy'},
              loss_weights={'dimension': 2., 'orientation': 1., 'confidence': 4.},
              metrics={'dimension': 'mse', 'orientation': 'mse', 'confidence': 'accuracy'}
             )

# get path of training samples
label_dir = 'training/label_2/'
image_dir = 'training/image_2/'

# map classes to integer indices
classes = [line.strip() for line in open(r'voc_labels.txt').readlines()]
cls_to_ind = {cls:i for i,cls in enumerate(classes)}

# get average dimensions of classes
dims_avg = np.loadtxt(r'voc_dims.txt',delimiter=',')

# process annotation files
objs = load_and_process_annotation_data(label_dir, dims_avg, cls_to_ind)
objs_num = len(objs)

# keep 90% as train set and shuffle
train_num = int(0.9*objs_num)
batch_size = 64
np.random.shuffle(objs)

# get training and validation samples after augmentation
train_gen = train_data_gen(objs[:train_num], image_dir, batch_size, bin_num=6)
valid_gen = train_data_gen(objs[train_num:], image_dir, batch_size, bin_num=6)

train_epoch_num = int(np.ceil((train_num)/batch_size))
valid_epoch_num = int(np.ceil((objs_num - train_num)/batch_size))

# start training
model.fit_generator(generator = train_gen,
                    steps_per_epoch = train_epoch_num,
                    epochs = 30,
                    verbose = 1,
                    validation_data = valid_gen,
                    validation_steps = valid_epoch_num,
                    callbacks = [early_stop, checkpoint],
                    max_queue_size = 3)

# save model
model.save_weights(r'weights.h5')