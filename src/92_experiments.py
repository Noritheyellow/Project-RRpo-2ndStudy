import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import multiprocessing
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model_src.DilatedResNet import DilatedResNet
from model_src.BianResnet import BianResNet
from model_src.LSTM import VanillaLSTM, CNNLSTM, BiLSTM, BiLSTMAttn
from model_src.RespNet import RespNet

print(f'Is GPU Avaliable: {tf.config.list_physical_devices("GPU")}')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
DATA_PATH = '/root/Workspace/DataLake/stMary'
DATA_SAVE_PATH = '/root/Workspace/Project-RRpo-2ndStudy/dataset' 

EPOCHS = 1000
BATCH_SIZE = 256
LR = 0.001
callbacks = [
    EarlyStopping(monitor='val_loss', patience=33),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
]

def gen_tfdataset(dataset, batchsize):
    X = []; y = []
    for pleth, resp in dataset:
        X.append(pleth.astype(np.float32))
        y.append(resp)

    X = np.array(X); y = np.array(y)
    scaler = MinMaxScaler()
    scaled_X = np.asarray([scaler.fit_transform(pleth.reshape(-1,1)) for pleth in X])
    print(f'Overall: {scaled_X.shape}, {y.shape}')
    return tf.data.Dataset.from_tensor_slices((scaled_X, y)).batch(batchsize)


def cross_validation(model, model_name, dataset, n_splits=5, batch_size=256, lr=0.001):
    kf = KFold(n_splits=n_splits)
    train_losses = []; val_losses = []
    subject_id = np.array([subject[0] for subject in dataset])
    count = 1
    
    for train_idx, val_idx in kf.split(subject_id):
        train_dataset = []; val_dataset = []
        
        for id, samples in dataset:
            if id in subject_id[train_idx]: train_dataset.extend(samples)
            else: val_dataset.extend(samples)
        
        train_dataset = np.array(train_dataset); val_dataset = np.array(val_dataset)

        train_tf = gen_tfdataset(train_dataset, batch_size)
        val_tf = gen_tfdataset(val_dataset, batch_size)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.MeanAbsoluteError(),
            metrics=keras.metrics.MeanAbsoluteError()
        )  

        callbacks.append(ModelCheckpoint(
            filepath=f'../models/230921/{model_name}/{model_name}-stmary-KF{count}/ckpt', 
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True))

        history = model.fit(
            train_tf,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=val_tf
        )

        callbacks.pop()

        min_val_loss_idx = np.argmin(history.history['val_loss'])
        train_losses.append(history.history['loss'][min_val_loss_idx])
        val_losses.append(history.history['val_loss'][min_val_loss_idx])
        count = count + 1
    
    print(f'TRAIN: {np.mean(train_losses)} ± {np.std(train_losses)}')
    print(f'VAL: {np.mean(val_losses)} ± {np.std(val_losses)}')
    return train_losses, val_losses


stmary = np.load(f'{DATA_SAVE_PATH}/230920/stmary-trainval_dataset.npy', allow_pickle=True)
print(stmary.shape)

models = [BiLSTM(units=64), BiLSTMAttn(units=64)]
model_names = ['BiLSTM', 'BiLSTMAttn']

results = np.array([(model_names[i], cross_validation(model, model_name=model_names[i], dataset=stmary, n_splits=5, batch_size=BATCH_SIZE, lr=LR)) for i, model in enumerate(models)])

print(results)