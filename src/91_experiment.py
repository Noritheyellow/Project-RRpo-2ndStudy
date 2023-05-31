import random
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
import keras
from scipy import signal
from itertools import starmap
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from models.Resnet import ResNet34
from models.Unet import Unet
from models.DilatedConv import RespDNN
from models.BianResnet import ResNet

DATA_PATH = '../../DataLake/stMary'
kf = KFold(n_splits=5)


def generate_dataset(arg_pleths, arg_resps, fs=125, shift_factor=4):
    import copy
    dataset = []
    window_size = fs * 60 # 7500
    shift = int(window_size/shift_factor) # 1875
    samples_len = len(arg_pleths)

    cpy_resps = copy.deepcopy(arg_resps)
    cpy_pleths = copy.deepcopy(arg_pleths)

    for i in range(samples_len):
        rr = cpy_resps[i]; ppg = cpy_pleths[i]

        rr['offset'] = (rr['offset']-rr['offset'].min())/1000
        size_lim = int(fs * np.ceil(rr['offset'].max()))
        ppg = ppg[:size_lim]
        shift_n_times = int((len(ppg)-window_size)/shift)+1

        samp_rr = [len(rr.loc[ (rr['offset']>=0+(int(shift/fs)*i)) & ((rr['offset']<int(window_size/fs)+(int(shift/fs)*i))) ]) for i in range(shift_n_times)]
        samp_ppg = [ppg[0+(shift*i):window_size+(shift*i)] for i in range(shift_n_times)]

        for i in range(len(samp_ppg)):
            temp = []
            temp.append(samp_ppg[i])
            temp.append(samp_rr[i])
            dataset.append(temp)

    return dataset


def interpolation(x, input):
    x0 = int(np.floor(x))
    y0 = input[x0]
    x1 = int(np.ceil(x))
    y1 = input[x1]
    y = (y1-y0)*(x-x0) + y0
    return y


def signal_resample(input_signal, org_fs, new_fs, method='interpolation'):
    output_signal = []
    new_x = np.arange(0, len(input_signal), org_fs/new_fs)
    
    if method == 'interpolation': 
        interp = interpolation

    for x in new_x:
        y = interp(x, input_signal)
        output_signal.append(y)

    return np.asarray(output_signal)


def preprocessing(targets=None):
    print('Extract PLETH/RESP')
    pleths = [pd.read_csv(f'{DATA_PATH}/{sid}/pleth.csv', header=None, names=['sid', 'offset', 'pleth']).pleth.values for sid in targets.id.unique()]
    resps = [pd.read_csv(f'{DATA_PATH}/{sid}/respirationTimeline.csv', header=None, names=['sid', 'offset']) for sid in targets.id.unique()]

    # Before filtering: Check NaN
    for pleth in pleths:
        if any(np.isnan(pleth)):
            print('check')

    # Before filtering: Convert type as np.int16
    pleths = list(map(lambda pleth: pleth.astype(np.int16), pleths))


    print('Init Preprocessing: Filtering')
    taps = signal.firwin(numtaps=400, cutoff=[0.5, 8.0], window='hamming', pass_zero=False, fs=125)
    w, h = signal.freqz(taps)
    pool = multiprocessing.Pool(processes=40)
    filtered_pleths = pool.starmap(signal.filtfilt, [(taps, 1.0, pleth) for pleth in pleths])
    pool.close()
    pool.join()


    print('Init Preprocessing: Windowing')
    dataset = generate_dataset(filtered_pleths, resps, shift_factor=60)


    print('Init Preprocessing: Resampling')
    pool = multiprocessing.Pool(processes=40)
    result = pool.starmap(signal_resample, [(pleth[0], 125, 30) for pleth in dataset])
    pool.close()
    pool.join()

    new_patient = []
    for i in range(len(dataset)):
        temp = []
        temp.append(result[i])
        temp.append(dataset[i][1])
        new_patient.append(temp)

    return new_patient


def prepare_modeling(dataset=None, batchsize=None):
    print(f'Prepare modeling')
    pleths = []
    resps = []
    for ppg, rr in dataset:
        pleths.append(ppg.astype(np.float32))
        resps.append(rr)
    pleths = np.asarray(pleths)
    resps = np.asarray(resps)
    print(pleths.shape, resps.shape)

    scaler = MinMaxScaler()
    scaled_pleths = np.asarray([scaler.fit_transform(pleth.reshape(-1,1)) for pleth in pleths])
    print(scaled_pleths.shape, type(scaled_pleths[0][0][0]))

    x, y = scaled_pleths[:], resps[:]

    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batchsize)


subjects = pd.read_csv(f'{DATA_PATH}/patients.csv')
patients = subjects.loc[subjects['diagnosis']!='0']

print('Init Sampling')
rand_idx = list(range(100))
random.seed(42)
test_idx = random.sample(rand_idx, k=20)
train_idx = list(set(rand_idx) - set(test_idx))
train_patients = patients.iloc[train_idx]
test_patients = patients.iloc[test_idx]
print('Train-Test Split into 80:20 (Random seed 42)')
print(train_idx, test_idx)


EPOCHS = 1000
BATCH_SIZE = 256
LR = 0.001
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]


counter = 1
dataset = {
    'train': [],
    'val': []
}
for train_idx, val_idx in kf.split(train_patients):
    print(f'{counter}th K-fold')
    counter = counter + 1
    # 이렇게 하는 이유는 Train과 Validation을 완벽히 구별시키기 위함이다.
    X_train = preprocessing(train_patients.iloc[train_idx])
    X_val = preprocessing(train_patients.iloc[val_idx])
    print(f'Preprocessing finished: {len(X_train)} / {len(X_val)}')
    
    train_dataset = prepare_modeling(X_train, batchsize=256)
    val_dataset = prepare_modeling(X_val, batchsize=256)

    dataset['train'].append(train_dataset)
    dataset['val'].append(val_dataset)



train_losses = []
val_losses = []
for i in range(5):
    print(f'{i+1}th K-fold')
    # model = ResNet34()
    model = ResNet()
    # model = Unet()
    # model = RespDNN()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=keras.metrics.MeanAbsoluteError()
    )  

    callbacks.append(ModelCheckpoint(f'../91_experiment_models/230531-Resnet-stmary-KF{i+1}', monitor='val_loss', save_best_only=True))
    
    history = model.fit(
        dataset['train'][i],
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=dataset['val'][i]
    )

    train_losses.append(min(history.history['loss']))
    val_losses.append(min(history.history['val_loss']))