from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D,
                          BatchNormalization, Activation, GaussianNoise, UpSampling2D, Add,
                          GlobalAveragePooling2D, Concatenate, Lambda, Conv2DTranspose, dot, Permute)

from keras import backend as K
from keras.regularizers import l2
import sys
import os
# from switch_norm import SwitchNormalization
from attention_layer import PositionAttentionLayer, ChannelAttentionModule

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))


def VggishConvBlock(input, filters, data_format, stddev=1.0, hasmaxpool=True, weight_decay=5e-4):
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    x = Conv2D(filters=filters, kernel_size=(3, 3), use_bias=False, padding='same', kernel_regularizer=l2(weight_decay),
               data_format=data_format)(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
               data_format=data_format, use_bias=False, kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    # x = Dropout(rate=0.2, seed=10)(x)
    if hasmaxpool:
        x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
        x = GaussianNoise(stddev=stddev)(x)
    return x


def ConvBlock(input, filters, data_format):
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    x = Conv2D(filters=filters, kernel_size=1, padding='same',
               data_format=data_format, use_bias=False, )(
        input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    return x


def Bottleneck(input, filters, stride=1, data_format='channels_first', activation='relu'):
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=stride, data_format=data_format, use_bias=False)(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation(activation)(x)

    return x


def Vggish(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'

    input_layer = Input(shape=(3, seq_len, mel_bins))

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format)
    x = VggishConvBlock(input=x, filters=128, data_format=data_format)
    x_1 = VggishConvBlock(input=x, filters=256, data_format=data_format)
    x = VggishConvBlock(input=x_1, filters=512, data_format=data_format)

    x = PositionAttentionLayer(512, data_format=data_format)(x)
    x = GlobalAveragePooling2D(data_format=data_format)(x)
    x = Dense(classes_num)(x)

    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def pooling_shape(input_shape):
    print(input_shape)
    if isinstance(input_shape, list):
        (sample_num, c, time_steps, freq_bins) = input_shape[0]
    else:
        (sample_num, c, time_steps, freq_bins) = input_shape

    return (sample_num, c)


def attention_pooling(input):
    x1, x2 = input
    x2 = K.clip(x2, K.epsilon(), 1 - K.epsilon())
    p = x2 / K.sum(x2, axis=[2, 3])[..., None, None]
    return K.sum(x1 * p, axis=[2, 3])


def attention_module(x, filters):
    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')
    x1 = Conv2D(filters=filters, kernel_size=(1, 1), data_format=data_format)(x)
    x1 = BatchNormalization(axis=bn_axis)(x1)
    x1 = Activation('sigmoid')(x1)
    x2 = Conv2D(filters=filters, kernel_size=(1, 1), data_format=data_format)(x)
    x2 = BatchNormalization(axis=bn_axis)(x2)
    x2 = Activation('softmax')(x2)
    x2 = Lambda(lambda x: K.log(x), )(x2)

    x = Lambda(attention_pooling, output_shape=pooling_shape, )([x1, x2])
    return x


def Vggish_danet(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'
    input_layer = Input(shape=(3, seq_len, mel_bins))

    weight_decay = 5e-4

    x = VggishConvBlock(input=input_layer, filters=64, data_format=data_format, weight_decay=weight_decay)
    x = VggishConvBlock(input=x, filters=128, data_format=data_format, weight_decay=weight_decay)
    x = VggishConvBlock(input=x, filters=256, data_format=data_format, weight_decay=weight_decay)
    x = VggishConvBlock(input=x, filters=512, data_format=data_format, weight_decay=weight_decay)

    x1 = ChannelAttentionModule()(x)
    x2 = PositionAttentionLayer(filters=K.int_shape(x)[1], weight_decay=weight_decay, data_format=data_format, )(x)

    # x = attention_module(Add()([x1, x2]), filters=64)
    # Bottleneck(x, 10)
    x = GlobalAveragePooling2D()(Add()([x1, x2]))
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_two_attention(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    input_layer = Input(shape=(3, seq_len, mel_bins))

    weight_decay = 5e-4

    x1 = VggishConvBlock(input=input_layer, filters=64, stddev=0.3, data_format=data_format, weight_decay=weight_decay)
    x2 = VggishConvBlock(input=x1, filters=128, stddev=0.3, data_format=data_format, weight_decay=weight_decay)
    x3 = VggishConvBlock(input=x2, filters=256, stddev=0.3, data_format=data_format, weight_decay=weight_decay)
    x4 = VggishConvBlock(input=x3, filters=512, stddev=0.3, data_format=data_format, weight_decay=weight_decay)

    deconv_x4 = Conv2DTranspose(256, 3, strides=2, padding='same', data_format=data_format, use_bias=False,
                                kernel_regularizer=l2(weight_decay))(x4)
    deconv_x4 = BatchNormalization(axis=bn_axis)(deconv_x4)
    deconv_x4 = Activation('relu')(deconv_x4)
    x_34 = Concatenate(axis=-1)([deconv_x4, x3])

    deconv_x3 = Conv2DTranspose(128, 3, strides=2, padding='same', data_format=data_format, use_bias=False,
                                kernel_regularizer=l2(weight_decay))(x3)
    deconv_x3 = BatchNormalization(axis=bn_axis)(deconv_x3)
    deconv_x3 = Activation('relu')(deconv_x3)
    x_23 = Concatenate(axis=-1)([deconv_x3, x2])

    x11 = attention_module(x_34, filters=64)
    x22 = attention_module(x_23, filters=32)

    x = Concatenate(axis=-1)([x11, x22])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def Vggish_three_attention(seq_len, mel_bins, classes_num):
    data_format = 'channels_first'
    if data_format == 'channels_first':
        bn_axis = 1
    else:
        raise Exception('Only support channels_first now!')

    input_layer = Input(shape=(2, seq_len, mel_bins))

    weight_decay = 5e-4

    x1 = VggishConvBlock(input=input_layer, filters=64, data_format=data_format, weight_decay=weight_decay)
    x2 = VggishConvBlock(input=x1, filters=128, data_format=data_format, weight_decay=weight_decay)
    x3 = VggishConvBlock(input=x2, filters=256, data_format=data_format, weight_decay=weight_decay)
    x4 = VggishConvBlock(input=x3, filters=512, data_format=data_format, weight_decay=weight_decay)

    deconv_x4 = Conv2DTranspose(256, 3, strides=2, padding='same', data_format=data_format, use_bias=False,
                                kernel_regularizer=l2(weight_decay))(x4)
    deconv_x4 = BatchNormalization(axis=bn_axis)(deconv_x4)
    deconv_x4 = Activation('relu')(deconv_x4)
    x_34 = Concatenate(axis=-1)([deconv_x4, x3])

    deconv_x3 = Conv2DTranspose(128, 3, strides=2, padding='same', data_format=data_format, use_bias=False,
                                kernel_regularizer=l2(weight_decay))(x3)
    deconv_x3 = BatchNormalization(axis=bn_axis)(deconv_x3)
    deconv_x3 = Activation('relu')(deconv_x3)
    x_23 = Concatenate(axis=-1)([deconv_x3, x2])

    deconv_x2 = Conv2DTranspose(64, 3, strides=2, padding='same', data_format=data_format, use_bias=False,
                                kernel_regularizer=l2(weight_decay))(x2)
    deconv_x2 = BatchNormalization(axis=bn_axis)(deconv_x2)
    deconv_x2 = Activation('relu')(deconv_x2)
    x_2 = Concatenate(axis=-1)([deconv_x2, x1])

    x11 = attention_module(x_34, filters=64)
    x22 = attention_module(x_23, filters=32)
    x33 = attention_module(x_2, filters=32)

    x = Concatenate(axis=-1)([x11, x22, x33])
    x = Dense(classes_num)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


if __name__ == '__main__':
    model = Vggish_two_attention(320, 64, 10)
    model.summary()
