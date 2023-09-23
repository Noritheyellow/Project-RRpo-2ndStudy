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
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten, Dropout, LSTM, Bidirectional, Concatenate, TimeDistributed
print(f'Is GPU Avaliable: {tf.config.list_physical_devices("GPU")}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore')

# 1st Linear
class VanillaLSTM(Model):
    def __init__(self, units=512, *args, **kwargs):
        super(VanillaLSTM, self).__init__(*args, **kwargs)
        self.lstm1 = LSTM(units=units, activation='tanh', return_sequences=True, return_state=False)
        self.lstm2 = LSTM(units=units//2, activation='tanh')
        self.d1 = Dense(1000, activation='relu')
        self.d2 = Dense(1000, activation='relu')
        self.d3 = Dense(1000, activation='relu')
        self.d4 = Dense(1)
    

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)


class CNNLSTM(Model):
    def __init__(self, init_filters=32, units=512, *args, **kwargs):
        super(CNNLSTM, self).__init__(*args, **kwargs)
        self.conv1 = Conv1D(filters=init_filters, kernel_size=3)
        self.bn1 = BatchNormalization()

        self.conv2 = Conv1D(filters=init_filters*2, kernel_size=3)
        self.bn2 = BatchNormalization()
        
        self.conv3 = Conv1D(filters=init_filters*4, kernel_size=3)
        self.bn3 = BatchNormalization()
        
        self.conv4 = Conv1D(filters=init_filters*8, kernel_size=3)
        self.bn4 = BatchNormalization()

        self.lstm1 = LSTM(units=units, activation='tanh')
        self.dense1 = Dense(1000, activation='relu')
        self.dense2 = Dense(1000, activation='relu')
        self.dense3 = Dense(1000, activation='relu')
        self.dense4 = Dense(1)


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = Activation('relu')(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = Activation('relu')(x)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = Activation('relu')(x)

        x = self.lstm1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)


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


class BiLSTM(tf.keras.models.Model):
    def __init__(self, units=128, *args, **kwargs):
        super(BiLSTM, self).__init__(*args, **kwargs)
        self.bilstm1 = Bidirectional(LSTM(units=units, activation='tanh', return_sequences=True))
        self.bilstm2 = Bidirectional(LSTM(units=units//2, activation='tanh'))
        self.d1 = Dense(1000, activation='relu')
        self.d2 = Dense(1000, activation='relu')
        self.d3 = Dense(1000, activation='relu')
        self.d4 = Dense(1)

    
    def call(self, inputs, training=None, mask=None):
        x = self.bilstm1(inputs)
        x = self.bilstm2(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)
    

class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)


    def call(self, values, query, training=None, mask=None):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))

        attention_weights = tf.nn.softmax(score, axis=1)

        context = attention_weights * values
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights

class BiLSTMAttn(Model):
    def __init__(self, units=128, units_attn=64, dropout=0.0):
        super(BiLSTMAttn, self).__init__()
        self.bilstm1 = Bidirectional(LSTM(units=units, dropout=dropout, return_sequences=True))
        self.bilstm2 = Bidirectional(LSTM(units=units//2, dropout=dropout, return_sequences=True, return_state=True))
        self.attention = BahdanauAttention(units=units_attn)
        self.d1 = Dense(1000, activation='relu')
        self.d2 = Dense(1000, activation='relu')
        self.d3 = Dense(1000, activation='relu')
        self.d4 = Dense(1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.bilstm1(inputs)
        x, forward_h, _, backward_h, _ = self.bilstm2(x)
        state_h = Concatenate()([forward_h, backward_h])
        context, attention_weights = self.attention(x, state_h)
        x = self.d1(context)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)

