# import the requirements
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
# dynamic usage of GPU RAM
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import backend as K
from keras import layers, models, optimizers

# K.set_image_data_format('channels_last')
import h5py
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate,GlobalAveragePooling2D
from keras import callbacks
import math
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
import json
from numpy.random import seed
import numpy as np

seed(1028)


######## Set the Parameters ##########
# 需要添加到config里面去的参数
# Number of mel-bins of the Magnitude Spectrogram
# melSize = 200

# Sub-Spectrogram Size
# splitSize = 20

# Mel-bins overlap

# overlap = 10
# seq_len = 320

# Time Indices
# timeInd = 500

# Channels used
# channels = 2


####### Generate the model ###########
# def subsp_net(channels, melSize, timeInd):
#     overlap = 10
#     splitSize = 20
#     seq_len = 320
#     data_format = 'channels_first'
#     bn_axis = 1
#     # outputs = []
#     toconcat = []
#     inputLayer = Input((channels, melSize, timeInd))
#     subSize = int(splitSize / 10)  # 2
#     i = 0
#     while (overlap * i <= melSize - splitSize):
#         # Create Sub-Spectrogram
#         INPUT = Lambda(lambda inputLayer: inputLayer[:, :, i * overlap:i * overlap + splitSize, :],
#                        output_shape=(channels, splitSize, timeInd))(inputLayer)
#         # print(0,INPUT.shape)
#
#         # First conv-layer -- 32 kernels
#         CONV = Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer="he_normal", data_format=data_format)(
#             INPUT)
#         # print(1,CONV.shape)
#
#         CONV = BatchNormalization(mode=0, axis=bn_axis,
#                                   gamma_regularizer=l2(0.0001),
#                                   beta_regularizer=l2(0.0001))(CONV)
#
#         # print(2,CONV.shape)
#         CONV = Activation('relu')(CONV)
#         # print(3,CONV.shape)
#         # Max pool by SubSpectrogram <mel-bin>/10 size. For example for sub-spec of 30x500, max pool by 3 vertically.
#         CONV = MaxPooling2D((2, 2), data_format=data_format)(CONV)
#         # CONV = Dropout(0.3)(CONV)
#         # print(4,CONV.shape)
#         # Second conv-layer -- 64 kernels
#         #CONV = Conv2D(64, kernel_size=(3, 3), padding='same',
#           #            kernel_initializer="he_normal", data_format=data_format)(CONV)
#         # print(5,CONV.shape)
#         #CONV = BatchNormalization(mode=0, axis=bn_axis,
#                                  # gamma_regularizer=l2(0.0001),
#                                  # beta_regularizer=l2(0.0001))(CONV)
#         # print(6,CONV.shape)
#         CONV = Activation('relu')(CONV)
#         # print(98,CONV.shape)
#         # Max pool
#         CONV = MaxPooling2D((2, 2), padding='same', data_format=data_format)(CONV)
#         # FLATTEN = tf.reshape(CONV, [20480])
#         # CONV = Dropout(0.30)(CONV)
#         # print(8, CONV.shape)  # (?,64,4,80)
#         # Flatten
#         #FLATTEN = Flatten()(CONV)  # 20480
#         # print(K.int_shape(FLATTEN),166,i)(None,20480)
#
#         # print(K.int_shape(FLATTEN))
#         ######################################问题所在#######################################################
#         # print(9,K.int_shape(FLATTEN)) K.int_shape求向量的长度192
#         #OUTLAYER = Dense(32, activation='relu')(FLATTEN)
#
#         OUTLAYER=GlobalAveragePooling2D(data_format=data_format)(CONV)
#         #print(OUTLAYER.shape)
#         # DROPOUT = Dropout(0.30)(OUTLAYER)
#         # print(i,K.int_shape(OUTLAYER))
#         # Sub-Classifier Layer
#         # FINALOUTPUT = Dense(10, activation='softmax')(OUTLAYER)
#         # print(11,FINALOUTPUT.shape)
#
#         # to be used for model output
#         #outputs.append(FINALOUTPUT)子分类器的输出
#         # 3print(12,outputs)
#         # array = numpy.array(outputs)
#         # print(117,array.shape)
#         # to be used for global classifier
#         toconcat.append(OUTLAYER)
#
#         # print(14,toconcat)
#         # array= numpy.array(toconcat)
#         # print(121,array.shape )
#
#         # print("--------------------",i,"---------------------")
#         i = i + 1
#
#     x = Concatenate()(toconcat)
#     #x=OUTLAYER
#     # print(x)#Tensor("concatenate_1/concat:0", shape=(?, 160), dtype=float32)
#     # print(15,x.shape)#(?,160)
#     # x = Concatenate(toconcat,axis=2)
#
#     # Automatically chooses appropriate number of hidden layers -- in a factor of 2.
#     # For example if  the number of sub-spectrograms is 9, we have 9*32 neurons by
#     # concatenating. So number of hidden layers would be 5 -- 512, 256, 128, 64
#     # numFCs = int(math.log(i * 32, 2))  # numFCs = 7
#     # neurons = math.pow(2, numFCs)  # neurous = 128
#     #
#     # Last layer before Softmax is a 64-neuron hidden layer
#     # while (neurons >= 64):
#     #     # i = 0
#     #     x = Dense(int(neurons), activation='relu')(x)
#     #     # print(i,x)
#     #     # print(i, x)
#     #     # x = Dropout(0.30)(x)
#     #     neurons = neurons / 2
#     # # i = i+1
#     # # neurons = 32
#     x = Dense(64, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)
#
#     # softmax -- GLOBAL CLASSIFIER
#     out = Dense(10, activation='softmax')(x)  # softmax最后输出是one-hot
#
#     # print(156,K.eval(out))
#     # outputs.append(out)
#
#     # Instantiate the model
#     classification_model = Model(inputLayer, out)
#     # print(classification_model)
#     return classification_model
#
#
# # # Summary
# # print(classification_model.summary())
# #
# #
# # # Compile the model
# # classification_model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
# #                              optimizer=Adam(lr=0.001),
# #                              metrics=['accuracy'])  # reporting the accuracy
# #
# # # Train the model
# # classification_model.fit(x_train, y_train, batch_size=16, epochs=200,
# #                          callbacks=[log, tb, checkpoint], verbose=1, validation_data=(x_test, y_test), shuffle=True)
# #
# # # classification_model.load_weights('resultsFinal/model_200.30.10.88.h5')
# # # y_pred = classification_model.predict(x_test)
# # # np.save('y_pred.npy', y_pred);
# if __name__ == '__main__':
# #     model = subsp_net(2, 64, 320)
# #     # tf.random_normal((32, 2, 64, 320))
# #     # model.predict()
#     model.summary()

from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D,
                          BatchNormalization, Activation, GaussianNoise, UpSampling2D, Add,
                          GlobalAveragePooling2D, Concatenate, Lambda, Conv2DTranspose, dot, Permute)

from keras import backend as K
from keras.layers import add, Flatten
from keras.metrics import top_k_categorical_accuracy
from keras.regularizers import l2
import sys
import os
# from switch_norm import SwitchNormalization
# from attention_layer import PositionAttentionLayer, ChannelAttentionModule

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):

    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name,
               data_format=data_format)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x

def bottleneck_Block(inpt, nb_filters, strides=(1, 1), with_conv_shortcut=False):

    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    k1, k2, k3 = nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet_50(channel, width, height, classes):
    inpt = Input(shape=(channel, width, height))
    x = ZeroPadding2D((3, 3), data_format='channels_first')(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_first')(x)

    # conv2_x
    x = bottleneck_Block(x, nb_filters=[64, 64, 256], strides=(1, 1), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64, 64, 256])
    x = bottleneck_Block(x, nb_filters=[64, 64, 256])

    # conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    # conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    # conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

if __name__ == '__main__':
    model = resnet_50(2, 320, 64, 10)
    model.summary()
