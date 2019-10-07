from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D,AveragePooling2D,ZeroPadding2D,
                          BatchNormalization, Activation, GaussianNoise, UpSampling2D, Add,
                          GlobalAveragePooling2D, Concatenate, Lambda, Conv2DTranspose, dot, Permute)

from keras import backend as K
from keras.layers import add, Flatten
from keras.metrics import top_k_categorical_accuracy
from keras.regularizers import l2
import sys
import os
# from switch_norm import SwitchNormalization
#from attention_layer import PositionAttentionLayer, ChannelAttentionModule

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

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name,data_format=data_format)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):

    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    k1,k2,k3=nb_filters
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

def resnet_50(channel,width,height,classes):
    inpt = Input(shape=(channel,width, height))
    x = ZeroPadding2D((3, 3),data_format='channels_first')(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',data_format='channels_first')(x)

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])

    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    #conv5_x
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
    model = resnet_50(4,320,64,10)
    model.summary()
