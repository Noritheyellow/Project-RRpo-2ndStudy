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
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten
print(f'Is GPU Avaliable: {tf.config.list_physical_devices("GPU")}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore')

DATA_PATH = '/root/Workspace/DataWarehouse/stMary_RRpo'


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


EPOCHS = 100
BATCH_SIZE = 256
LR = 0.001
kf = KFold(n_splits=5)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    ModelCheckpoint('../models/230522-Resnet', monitor='val_loss', save_best_only=True)
]

model = ResNet34()
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, weight_decay=0.0001),
    loss=keras.losses.MeanAbsoluteError(),
    metrics=keras.metrics.MeanAbsoluteError()
)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(BATCH_SIZE)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_dataset
)


min(history.history['val_loss'])