import gzip
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import os
import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten, Dropout
print(f'Is GPU Avaliable: {tf.config.list_physical_devices("GPU")}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.style.use('ggplot')

DATA_PATH = '/root/Workspace/DataWarehouse/stMary_RRpo'

class RespBlock(Model):
    def __init__(self, filters, *args, **kwargs):
        super(RespBlock, self).__init__(*args, **kwargs)
        self.conv1 = Conv1D(filters=filters, kernel_size=3, strides=1, dilation_rate=1, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(filters=filters, kernel_size=3, strides=1, dilation_rate=2, padding='same')
        self.bn2 = BatchNormalization()
        self.conv3 = Conv1D(filters=filters, kernel_size=3, strides=1, dilation_rate=3, padding='same')
        self.bn3 = BatchNormalization()
        self.conv4 = Conv1D(filters=filters, kernel_size=3, strides=1, dilation_rate=5, padding='same')
        self.bn4 = BatchNormalization()
        self.conv1x1 = Conv1D(filters=filters*2, kernel_size=1, strides=1, padding='same')
        self.bn1x1 = BatchNormalization()


    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1)
        x1 = Activation('relu')(x1)
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2)
        x2 = Activation('relu')(x2)
        x3 = self.conv3(inputs)
        x3 = self.bn3(x3)
        x3 = Activation('relu')(x3)
        x4 = self.conv4(inputs)
        x4 = self.bn4(x4)
        x4 = Activation('relu')(x4)

        x = Add()([x1, x2, x3, x4])
        x = self.conv1x1(x)
        x = self.bn1x1(x)

        if inputs.shape[-1] != 1:
            inputs = tf.reduce_mean(inputs, axis=-1, keepdims=True)
           
        x = Add()([x, inputs])

        return x



class RespDNN(Model):
    def __init__(self, *args, **kwargs):
        super(RespDNN, self).__init__(*args, **kwargs)
        self.respblk = [RespBlock(32*i) for i in np.arange(1, 4)]
        self.dwnsamp = [Conv1D(32*i, kernel_size=3, strides=2, padding='same') for i in np.arange(1, 4)]
        self.bn = [BatchNormalization() for _ in range(3)]
        self.avgpool = AveragePooling1D(strides=2, padding='valid')
        self.dense1 = Dense(1000, activation='relu')
        self.dense2 = Dense(100, activation='relu')
        self.dense3 = Dense(1)

    
    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(3):
           x = self.respblk[i](x)
           x = self.dwnsamp[i](x)
           x = self.bn[i](x)
           x = Activation('relu')(x)

        x = self.avgpool(x)
        x = Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


with gzip.open(f'{DATA_PATH}/21_230531_resamp_sliced125_filt_stmary_train_patients.pickle.gzip', 'rb') as f:
    dataset = pickle.load(f)

random.seed(42)
random.shuffle(dataset)

pleths = []
resps = []
for ppg, rr in dataset:
    pleths.append(ppg.astype(np.float64))
    resps.append(rr)

pleths = np.asarray(pleths)
resps = np.asarray(resps)
print(pleths.shape, resps.shape)

scaler = MinMaxScaler()
scaled_pleths = np.asarray([scaler.fit_transform(pleth.reshape(-1,1)) for pleth in pleths])
print(scaled_pleths.shape, type(scaled_pleths[0][0][0]))

ratio_tr = 0.8
train_x, train_y = scaled_pleths[:int(len(scaled_pleths)*ratio_tr)], resps[:int(len(resps)*ratio_tr)]
val_x, val_y = scaled_pleths[int(len(scaled_pleths)*ratio_tr):], resps[int(len(resps)*ratio_tr):]
print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)


EPOCHS = 100000
BATCH_SIZE = 256
LR = 0.001
kf = KFold(n_splits=5)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    ModelCheckpoint('../models/230531-RespDNN-3times-4dil-stmary', monitor='val_loss', save_best_only=True)
]

model = RespDNN()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    # optimizer=tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, weight_decay=0.0001),
    loss=keras.losses.MeanAbsoluteError(),
    metrics=keras.metrics.MeanAbsoluteError()
)


train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(BATCH_SIZE)

with tf.device('/GPU:0'):
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=val_dataset
    )