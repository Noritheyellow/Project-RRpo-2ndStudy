import gzip
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from utils.signal_processing import *
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import os
import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten, Dropout, Concatenate
print(f'Is GPU Avaliable: {tf.config.list_physical_devices("GPU")}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore')

DATA_PATH = '/root/Workspace/DataWarehouse/stMary_RRpo'

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
        x = inputs # 1800

        x, feature_map1 = self.contraction_blocks[0](x) # 32->64 900
        x, feature_map2 = self.contraction_blocks[1](x) # 64->128 450
        x, feature_map3 = self.contraction_blocks[2](x) # 128->256 225
        # x, feature_map4 = self.contraction_blocks[3](x) # 256->512 

        x = self.bottleneck_block(x)

        x = self.expansion_blocks[1](x, feature_map3) # 256->128 
        x = self.expansion_blocks[2](x, feature_map2)
        x = self.expansion_blocks[3](x, feature_map1)
        # x = self.expansion_blocks[3](x, feature_map1)


        x = self.avgpool(x)
        x = Flatten()(x)
        x = self.d100(x)
        x = self.d50(x)
        x = self.d10(x)
        
        return self.d1(x)


with gzip.open(f'{DATA_PATH}/21_230518_resamp_sliced125_filt_patient_stmary.pickle.gzip', 'rb') as f:
    dataset = pickle.load(f)

print(len(dataset), len(dataset[0][0]))

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


EPOCHS = 1000
BATCH_SIZE = 256
LR = 0.001
kf = KFold(n_splits=5)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    ModelCheckpoint('../models/230531-Unet-B256-SGD', monitor='val_loss', save_best_only=True)
]

model = Unet()
model.compile(
    # optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    optimizer=tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, weight_decay=0.0001),
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