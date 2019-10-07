from keras.layers import Layer, Conv2D, Softmax, Reshape, Permute, BatchNormalization, Activation
from keras import initializers
from keras import regularizers
import keras.backend as k


class BottleNeck(Layer):
    def __init__(self, filters, weight_decay, data_format):
        if data_format == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        self.conv = Conv2D(filters=filters, kernel_size=1, use_bias=False, padding='same',
                           kernel_regularizer=regularizers.l2(weight_decay), data_format=data_format)
        self.bn = BatchNormalization(axis=bn_axis)
        self.activation = Activation('relu')
        super(BottleNeck, self).__init__()

    def call(self, x, **kwargs):
        return self.activation(self.bn(self.conv(x)))


class PositionAttentionLayer(Layer):
    # 自定义层 reshape permute 指定的shape 不考虑batch维度!！!！!！!
    # batch_dot 注意 x_batch,y_batch的顺序

    def __init__(self, filters, weight_decay, data_format, **kwargs):
        self.filters = filters
        self.gamma = None
        self.kernel_initializer = initializers.Zeros()
        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.query_conv = BottleNeck(80, weight_decay=weight_decay, data_format=data_format)
        self.key_conv = BottleNeck(80, weight_decay=weight_decay, data_format=data_format)
        self.value_conv = BottleNeck(filters, weight_decay=weight_decay, data_format=data_format)
        self.softmax = Softmax(axis=-1)

        super(PositionAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        # 这里的超参数gamma是系数权重,和输入维度一致 shape也不考虑batch维度
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer=self.kernel_initializer,
                                     trainable=True, )

        super(PositionAttentionLayer, self).build(input_shape)  # set build=true

    def call(self, x, **kwargs):
        _, C, H, W = x.shape

        def hw_flatten(x):
            return Reshape((-1, H * W))(x)

        proj_query = Permute(dims=(2, 1))(hw_flatten(self.query_conv(x)))
        proj_key = hw_flatten(self.key_conv(x))
        energy = k.batch_dot(proj_query, proj_key, axes=[1, 2])
        attention = self.softmax(energy)

        proj_value = hw_flatten(self.value_conv(x))
        attention = Permute(dims=(2, 1))(attention)
        out = k.batch_dot(attention, proj_value, axes=[1, 2])
        out = Reshape((C, H, W))(out)
        out = self.gamma * out + x
        return out

    def compute_output_shape(self, input_shape):
        print('compute_output_shape：', input_shape)
        return input_shape


class ChannelAttentionModule(Layer):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = None
        self.softmax = Softmax(axis=-1)

    def call(self, x, **kwargs):
        _, C, H, W = x.shape

        def hw_flatten(x):
            return Reshape((-1, H * W))(x)

        query = hw_flatten(x)
        key = Permute(dims=(2, 1))(hw_flatten(x))
        energy = k.batch_dot(query, key, axes=[1, 2])
        attention = self.softmax(energy)

        value = hw_flatten(x)
        out = k.batch_dot(attention, value, axes=[1, 2])
        out = Reshape((-1, H, W))(out)
        out = self.gamma * out + x

        return out

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer=initializers.Zeros(),
                                     trainable=True)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape
