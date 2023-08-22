import os
import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten, Dropout, Concatenate


class ContractionBlock(Model):
    def __init__(self, starting_filter=32, dropout=0, *args, **kwargs):
        super(ContractionBlock, self).__init__(*args, **kwargs)
        self.conv1a = Conv1D(filters=starting_filter, kernel_size=3, strides=1, padding='same')
        self.conv1b = Conv1D(filters=starting_filter*2, kernel_size=3, strides=1, padding='same')
        self.maxpool = MaxPooling1D(pool_size=2, strides=2, padding='same')
        
        if dropout!=0:
            self.dropout = Dropout(rate=dropout)
        else:
            self.dropout = None


    def call(self, inputs, training=None, mask=None):
        feature_map = self.conv1a(inputs)
        feature_map = Activation('relu')(feature_map)
        feature_map = self.conv1b(feature_map)
        feature_map = Activation('relu')(feature_map)
        
        if self.dropout is not None:
            feature_map = self.dropout(feature_map)

        outputs = self.maxpool(feature_map)

        return outputs, feature_map
    

class BottleNeckBlock(Model):
    def __init__(self, starting_filter=512, dropout=0, *args, **kwargs):
        super(BottleNeckBlock, self).__init__(*args, **kwargs)
        self.conv1a = Conv1D(filters=starting_filter, kernel_size=3, strides=1, padding='same')
        self.conv1b = Conv1D(filters=starting_filter*2, kernel_size=3, strides=1, padding='same')
        self.dropout = Dropout(dropout)


    def call(self, inputs, training=None, mask=None):
        outputs = self.conv1a(inputs)
        outputs = Activation('relu')(outputs)
        outputs = self.conv1b(outputs)
        outputs = Activation('relu')(outputs)
        outputs = self.dropout(outputs)
        
        return outputs
    

class ExpansionBlock(Model):
    def __init__(self, starting_filter=512, *args, **kwargs):
        super(ExpansionBlock, self).__init__(*args, **kwargs)
        self.conv1a = Conv1D(filters=starting_filter*2, kernel_size=3, strides=1, padding='same')
        self.conv1b = Conv1D(filters=starting_filter, kernel_size=3, strides=1, padding='same')
        self.upsampling = Conv1DTranspose(filters=starting_filter, kernel_size=2, strides=2, padding='same')
    

    def call(self, inputs, feature_map, training=None, mask=None):
        x = self.upsampling(inputs)
        x = Concatenate(axis=2)([x, feature_map])
        x = self.conv1a(x)
        x = Activation('relu')(x)
        x = self.conv1b(x)
        x = Activation('relu')(x)

        return x
    

class Unet(Model):
    def __init__(self, *args, **kwargs):
        super(Unet, self).__init__(*args, **kwargs)
        self.contraction_blocks = [ContractionBlock(starting_filter=f, dropout=d) for f, d in [(32,0), (64,0), (128,0), (256,0.5)]]
        self.bottleneck_block = BottleNeckBlock(starting_filter=256, dropout=0.5)
        self.expansion_blocks = [ExpansionBlock(starting_filter=f) for f in [512, 256, 128, 64]]
        self.avgpool = AveragePooling1D(pool_size=2, strides=1, padding='valid')

        self.d100 = Dense(100, activation='relu')
        self.d50 = Dense(50, activation='relu')
        self.d10 = Dense(10, activation='relu')
        self.d1 = Dense(1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = inputs

        x, feature_map1 = self.contraction_blocks[0](x)
        x, feature_map2 = self.contraction_blocks[1](x)
        x, feature_map3 = self.contraction_blocks[2](x)
        # x, feature_map4 = self.contraction_blocks[3](x)

        x = self.bottleneck_block(x)

        x = self.expansion_blocks[1](x, feature_map3)
        x = self.expansion_blocks[2](x, feature_map2)
        x = self.expansion_blocks[3](x, feature_map1)
        # x = self.expansion_blocks[3](x, feature_map1)


        x = self.avgpool(x)
        x = Flatten()(x)
        x = self.d100(x)
        x = self.d50(x)
        x = self.d10(x)
        
        return self.d1(x)