import gzip
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, Add, LeakyReLU, MaxPooling1D, Flatten, Dense, BatchNormalization, Activation
print(f'Is GPU Avaliable: {tf.config.list_physical_devices("GPU")}')

DATA_PATH = '/root/Workspace/DataWarehouse/stMary_RRpo'


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
    
    
with gzip.open(f'{DATA_PATH}/21_230518_resamp_sliced125_filt_patient_stmary.pickle.gzip', 'rb') as f:
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


EPOCHS = 1000
BATCH_SIZE = 256
LR = 0.001
kf = KFold(n_splits=5)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    ModelCheckpoint('../models/230525-BianResnet-SGD', monitor='val_loss', save_best_only=True)
]


# optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
model = ResNet()
model.compile(
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