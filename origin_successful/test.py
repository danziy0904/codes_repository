# import h5py
# import numpy as np
#
#
# labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
#           'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
#
# lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
# ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}
#
#
#
# hdf5_path = "/home/r506/hhy/workspace/features/logmel/TUT-urban-acoustic-scenes-2018-development/development_delta11.h5"
# #hdf5_path = "/home/r506/wq/workspace/features/logmel/TUT-urban-acoustic-scenes-2018-development/development_lr_hp.h5"
# hf = h5py.File(hdf5_path, 'r')
#
# audio_names = np.array([s.decode() for s in hf['filename'][:]])
# x = hf['feature'][:]
# scene_labels = [s.decode() for s in hf['scene_label'][:]]
# identifiers = [s.decode() for s in hf['identifier'][:]]
# source_labels = [s.decode() for s in hf['source_label']]
# y = np.array([lb_to_ix[lb] for lb in scene_labels])
#
# print(audio_names)
# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))
import numpy
# import tensorflow as tf
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# print(sess.run(c))
# import tensorflow as tf
# version = tf.__version__
# gpu_ok = tf.test.is_gpu_available()
# print("tf version:",version,"\nuse GPU",gpu_ok)
#
import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D,
                          BatchNormalization, Activation, GaussianNoise, UpSampling2D, Add,
                          GlobalAveragePooling2D, Concatenate, Lambda, Conv2DTranspose, dot, Permute)

from keras import backend as K
from keras.regularizers import l2
import sys
import os

# x11=np.array([[1,2,5,6],[5,6,5,6]])
# x22=np.array([[2,3,8,9],[3,4,4,6]])
#
# x11= tf.convert_to_tensor(x11)
# x22= tf.convert_to_tensor(x22)
# x = Concatenate(axis=-1)([x11, x22])
# # print(x.shape)
# # print(x11)
# #x111=np.array(x)
#
#
# print(x.type)
# # print(x111.shape)
# # # print(x111)