import os
import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten, Dropout


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