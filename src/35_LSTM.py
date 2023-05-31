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
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten, Dropout, LSTM, Bidirectional, Concatenate
print(f'Is GPU Avaliable: {tf.config.list_physical_devices("GPU")}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.style.use('ggplot')

DATA_PATH = '/root/Workspace/DataWarehouse/stMary_RRpo'

# 1st Linear
class VanillaLSTM(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm1 = LSTM(15)
        self.d1 = Dense(1)
    
    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.lstm1(x)
        x = Flatten()(x)
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
    ModelCheckpoint('../models/230525-VanillaLSTM', monitor='val_loss', save_best_only=True)
]

model = VanillaLSTM()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
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