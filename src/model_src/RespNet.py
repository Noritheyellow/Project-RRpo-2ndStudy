import os
import keras
import warnings
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, AveragePooling1D, Dense, BatchNormalization, Activation, Add, Flatten, Dropout, Concatenate, LeakyReLU
print(f'Is GPU Avaliable: {tf.config.list_physical_devices("GPU")}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore')

DATA_PATH = '/root/Workspace/DataWarehouse/stMary_RRpo'

class DResIncBlock(Model):
    def __init__(self, starting_filter=32, dropout=0, *args, **kwargs):
        super(DResIncBlock, self).__init__(*args, **kwargs)
        self.conv11 = Conv1D(filters=starting_filter, kernel_size=1, strides=1, padding='same')
        self.bn11 = BatchNormalization()

        self.conv21 = Conv1D(filters=starting_filter, kernel_size=1, strides=1, padding='same')
        self.bn21 = BatchNormalization()
        self.conv22 = Conv1D(filters=starting_filter, kernel_size=3, strides=1, dilation_rate=2, padding='same')
        self.bn22 = BatchNormalization()

        self.conv31 = Conv1D(filters=starting_filter, kernel_size=1, strides=1, padding='same')
        self.bn31 = BatchNormalization()
        self.conv32 = Conv1D(filters=starting_filter, kernel_size=3, strides=1, dilation_rate=4, padding='same')
        self.bn32 = BatchNormalization()

        self.conv41 = Conv1D(filters=starting_filter, kernel_size=1, strides=1, padding='same')
        self.bn41 = BatchNormalization()
        self.conv42 = Conv1D(filters=starting_filter, kernel_size=3, strides=1, dilation_rate=8, padding='same')
        self.bn42 = BatchNormalization()


    def call(self, inputs, training=None, mask=None):
        # inputs ?, timestep, features
        x1 = self.conv11(inputs) 
        x1 = self.bn11(x1)
        
        x2 = self.conv21(inputs)
        x2 = self.bn21(x2)
        x2 = Activation('relu')(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        
        x3 = self.conv31(inputs)
        x3 = self.bn31(x3)
        x3 = Activation('relu')(x3)
        x3 = self.conv32(x3)
        x3 = self.bn32(x3)
        
        x4 = self.conv41(inputs)
        x4 = self.bn41(x4)
        x4 = Activation('relu')(x4)
        x4 = self.conv42(x4)
        x4 = self.bn42(x4)
        
        x = Concatenate()([x1, x2, x3, x4]) # ?, input_timestep, features * 4
        outputs = Add()([x, inputs])
        return outputs


class ContractingPath(Model):
    def __init__(self, filters, *args, **kwargs):
        super(ContractingPath, self).__init__(*args, **kwargs)
        self.blocks = [DResIncBlock(starting_filter=filters*(2**i)) for i in range(3)]
        self.dwnconv = [Conv1D(filters=1, kernel_size=2, strides=2, padding='same') for _ in range(3)]
        self.bn = [BatchNormalization() for _ in range(3)]


    def call(self, inputs, training=None, mask=None):
        f_map1 = self.blocks[0](inputs) # ?, 1800, 128(64)
        x1 = self.dwnconv[0](f_map1) 
        x1 = self.bn[0](x1)
        x1 = LeakyReLU(alpha=0.2)(x1) # ?, 900, 128(64)

        f_map2 = self.blocks[1](x1) # ?, 900, 256(128)
        x2 = self.dwnconv[1](f_map2)
        x2 = self.bn[1](x2)
        x2 = LeakyReLU(alpha=0.2)(x2) # ?, 450, 256(128)

        f_map3 = self.blocks[2](x2) # ?, 450, 512(256)
        x3 = self.dwnconv[2](f_map3)
        x3 = self.bn[2](x3)
        x3 = LeakyReLU(alpha=0.2)(x3) # ?, 225, 512(256)

        return x3, f_map1, f_map2, f_map3
    

class ExpandingPath(Model):
    def __init__(self, filters=512, *args, **kwargs):
        super(ExpandingPath, self).__init__(*args, **kwargs)
        self.upconv = [Conv1DTranspose(filters=int(filters/(2**i)), kernel_size=2, strides=2, padding='same') for i in range(3)]
        self.blocks = [DResIncBlock(starting_filter=int((filters/(2**i))/4)) for i in range(3)]
        self.bn = [BatchNormalization() for _ in range(3)]
    

    def call(self, inputs, feature_map, training=None, mask=None):
        # (?, 225, 512)
        x = self.upconv[2](inputs) # (?, 450, 512)
        x = self.bn[2](x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.blocks[2](x) # (?, 450, 512)
        x = Concatenate()([x, feature_map[2]]) # (?, 450, 1024)

        x = self.upconv[1](x) # (?, 900, 256)
        x = self.bn[1](x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.blocks[1](x) # (?, 900, 256)
        x = Concatenate()([x, feature_map[1]]) # (?, 900, 512)

        x = self.upconv[0](x) # (?, 1800, 128)
        x = self.bn[0](x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.blocks[0](x) # (?, 1800, 128)
        x = Concatenate()([x, feature_map[0]]) # (?, 1800, 256)

        return x
    

class RespNet(Model):
    def __init__(self, *args, **kwargs):
        super(RespNet, self).__init__(*args, **kwargs)
        self.cpath = ContractingPath(filters=16)
        self.bottleneck = DResIncBlock(starting_filter=64)
        self.xpath = ExpandingPath(filters=256)
        self.conv1 = Conv1D(filters=1, kernel_size=2, strides=1, padding='valid')
        self.bn1 = BatchNormalization()

        self.d100 = Dense(100, activation='relu')
        self.d50 = Dense(50, activation='relu')
        self.d10 = Dense(10, activation='relu')
        self.d1 = Dense(1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = inputs # ?, 1800, 1
        x, f1, f2, f3 = self.cpath(x) # (?, 225, 512), (?, 1800, 128), (?, 900, 256), (?, 450, 512)
        x = self.bottleneck(x) # (?, 225, 512)
        x = self.xpath(x, (f1, f2, f3)) # (?, 1800, 256)

        x = self.conv1(x) # (?, 1800, 1) # 여기가 문제인 듯? 위에 x는 샘플?의 2차원 행렬일텐데 그걸 1차원 conv로 풀려고 하면 256번 하겠지, stride=2에 valid니까 timestep은 1799일테고,
        x = self.bn1(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)

        x = self.d100(x)
        x = self.d50(x)
        x = self.d10(x)

        return self.d1(x)