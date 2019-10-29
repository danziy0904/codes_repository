

from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D,
                          BatchNormalization, Activation, GaussianNoise, UpSampling2D, Add,
                          GlobalAveragePooling2D, Concatenate, Lambda, Conv2DTranspose, dot, Permute)

from keras import backend as K
from keras.regularizers import l2
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
import json
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

def ord_net(channels, mel_bins, time_Id, data_format):
    # 网络输入尺寸（3，64，320）
    if data_format == "channels_first":
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')
    inputLayer = Input((channels, mel_bins, time_Id))  # 输入尺寸
    x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False, padding='same', data_format=data_format)(inputLayer)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False, padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    # x = GlobalAveragePooling2D(data_format=data_format)(x)
    x = Dense(32, activation='softmax')(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputLayer, outputs=x)

    return model

if __name__ == '__main__':
    model = ord_net(3, 64, 320, 'channels_first')
    model.summary()






