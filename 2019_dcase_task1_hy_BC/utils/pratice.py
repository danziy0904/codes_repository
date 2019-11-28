import tensorflow as tf

tf.enable_eager_execution()

a = tf.random_uniform((10, 3, 2, 2))
b = tf.nn.softmax(a, dim=-1)
c = tf.reduce_mean(b, axis=-1)
print(c)
