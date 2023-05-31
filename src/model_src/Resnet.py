import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten


class ResidualBlock(Model):
    def __init__(self, filters, kernel_size, strides, identity_mapping=None, *args, **kwargs):
        super(ResidualBlock, self).__init__(*args, **kwargs)
        self.conv1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides[0], padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides[1], padding='same')
        self.bn2 = BatchNormalization()

        self.identity_mapping = identity_mapping
        self.conv_identity = Conv1D(filters=filters, kernel_size=1, strides=strides[0], padding='same')
        

    def call(self, inputs, training=None, mask=None):
        identity = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # 448, 64 / 224, 128
        if self.identity_mapping:
            identity = self.conv_identity(inputs)
            # print(inputs.shape, identity.shape)

        x = Add()([x, identity])
        return Activation('relu')(x)


class ResNet34(Model):
    def __init__(self, *args, **kwargs):
        super(ResNet34, self).__init__(*args, **kwargs)
        # self.conv1 = Conv1D(filters=64, kernel_size=7, strides=2)
        # self.max1d = MaxPooling1D(pool_size=3, strides=2)
        self.resnet_block1 = [ResidualBlock(64, 3, (1,1)) for i in range(3)]
        
        self.resnet_block2_entry = ResidualBlock(128, 3, (2,1), identity_mapping=True)
        self.resnet_block2 = [ResidualBlock(128, 3, (1,1)) for i in range(3)]

        self.resnet_block3_entry = ResidualBlock(256, 3, (2,1), identity_mapping=True)
        self.resnet_block3 = [ResidualBlock(256, 3, (1,1)) for i in range(5)]

        self.resnet_block4_entry = ResidualBlock(512, 3, (2,1), identity_mapping=True)
        self.resnet_block4 = [ResidualBlock(512, 3, (1,1)) for i in range(2)]

        self.avg1d = AveragePooling1D(strides=2, padding='same')
        self.flatten = Flatten()
        self.d100 = Dense(100, activation='relu')
        self.d50 = Dense(50, activation='relu')
        self.d10 = Dense(10, activation='relu')
        self.d1 = Dense(1)

    
    def call(self, inputs, training=None, mask=None):
        # x = self.conv1(inputs)
        # x = self.max1d(x)

        x = inputs
        for block in self.resnet_block1:
            x = block(x, training=training)

        x = self.resnet_block2_entry(x, training=training)
        for block in self.resnet_block2:
            x = block(x, training=training)

        x = self.resnet_block3_entry(x, training=training)
        for block in self.resnet_block3:
            x = block(x, training=training)
        
        x = self.resnet_block4_entry(x, training=training)
        for block in self.resnet_block4:
            x = block(x, training=training)
        
        
        x = self.avg1d(x)
        x = self.flatten(x)
        x = self.d100(x)
        x = self.d50(x)
        x = self.d10(x)
        return self.d1(x)
    
    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        x, y = data

        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}