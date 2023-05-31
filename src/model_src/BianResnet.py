import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, Add, LeakyReLU, MaxPooling1D, Flatten, Dense, BatchNormalization, Activation


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ResnetIdentityBlock, self).__init__(name='resnet_block')
        self.conv1a = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.bn1a = BatchNormalization()

        self.conv1b = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')
        self.bn1b = BatchNormalization()

        self.conv1c = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')
        self.bn1c = BatchNormalization()

    
    def call(self, input, training=False):
        x0 = self.conv1a(input)
        x0 = self.bn1a(x0, training=training)
        x0 = Activation('relu')(x0)

        x1 = self.conv1b(x0)
        x1 = self.bn1b(x1, training=training)
        x1 = Activation('relu')(x1)
        x1 = self.conv1c(x1)
        x1 = self.bn1c(x1, training=training)
        x1 = Activation('relu')(x1)

        x = Add()([x0 + x1])
        return Activation('relu')(x)


class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet_block = [ResnetIdentityBlock(6*(2**i), 3, 2) for i in range(5)]
        self.max1d = MaxPooling1D(strides=2, padding='same')
        self.flatten = Flatten()
        self.d20 = Dense(20, activation='relu')
        self.d10 = Dense(10, activation='relu')
        self.d1 = Dense(1)

    
    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(5):
            x = self.resnet_block[i](x, training=training)

        x = self.max1d(x)
        x = self.flatten(x)
        x = self.d20(x)
        x = self.d10(x)
        return self.d1(x)
    

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