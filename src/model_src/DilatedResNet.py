import os
import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, Dense, BatchNormalization, Activation, Add, Flatten, Concatenate


class RespBlock(Model):
    def __init__(self, filters, kernel_size, dilation_rate, *args, **kwargs):
        super(RespBlock, self).__init__(*args, **kwargs)
        self.conv11 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')
        self.bn11 = BatchNormalization()

        self.conv12 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')
        self.bn12 = BatchNormalization()

        self.conv2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate+1, padding='same')
        self.bn2 = BatchNormalization()

        self.conv3 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate+2, padding='same')
        self.bn3 = BatchNormalization()


    def call(self, inputs, training=None, mask=None):
        x1 = self.conv11(inputs)
        x1 = self.bn11(x1, training=training)
        x1 = Activation('relu')(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12(x1, training=training)
        x1 = Activation('relu')(x1)

        x2 = self.conv2(inputs)
        x2 = self.bn2(x2, training=training)
        x2 = Activation('relu')(x2)
        
        x3 = self.conv3(inputs)
        x3 = self.bn3(x3, training=training)
        x3 = Activation('relu')(x3)

        x = Add()([x1, x2, x3])
        x = Activation('relu')(x)

        if inputs.shape[-1] != 1:
            inputs = tf.reduce_mean(inputs, axis=-1, keepdims=True)
           
        x = Add()([x, inputs])
        return Activation('relu')(x)


class DilatedResNet(Model):
    def __init__(self, num_of_blocks=2, kernel_size=3, dilation_rate=1, dwn_kernel_size=2, filters=8, units=100, *args, **kwargs):
        super(DilatedResNet, self).__init__(*args, **kwargs)
        self.num_of_blocks = num_of_blocks
        self.respblk = [RespBlock(filters*(2**i), kernel_size=kernel_size, dilation_rate=dilation_rate) for i in range(num_of_blocks)]
        self.dwnsamp = [Conv1D(filters*(2**i), kernel_size=dwn_kernel_size, strides=2, padding='same') for i in range(num_of_blocks)]
        self.bn = [BatchNormalization() for _ in range(num_of_blocks)]
        self.avgpool = AveragePooling1D(strides=2, padding='valid')
        self.dense1 = Dense(units=units, activation='relu') #1000
        self.dense2 = Dense(10, activation='relu') #100
        self.dense3 = Dense(1)


    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(self.num_of_blocks):
            x = self.respblk[i](x, training=training)
            x = self.dwnsamp[i](x)
            x = self.bn[i](x, training=training)
            x = Activation('relu')(x)

        x = self.avgpool(x)
        x = Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    

    def test_step(self, data):
        x, y = data

        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}