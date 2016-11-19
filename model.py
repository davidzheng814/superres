import tensorflow as tf
import numpy as np
import glob
import argparse

BN_EPSILON = 0.001

class Loader(object):
    def __init__(self, high_res_info):
        f2, self.h2, self.w2 = high_res_info
        self.q2 = tf.train.string_input_producer(f2)
        self.batch_size = 50

    def _get_pipeline(self, q, h, w):
        reader = tf.WholeFileReader()
        key, value = reader.read(q)
        raw_img = tf.image.decode_png(value, channels=3)
        my_img = tf.image.per_image_whitening(raw_img)
        my_img.set_shape((h, w, 3))
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * self.batch_size
        batch = tf.train.shuffle_batch([my_img], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue)
        small_batch = tf.images.resize_bicubic(batch, [96, 96])
        return small_batch, batch

    def batch():
        return self._get_pipeline(self.q2, self.h2, self.w2)

def relu(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                             tf.cast(max_value, dtype=_FLOATX))
    x -= tf.constant(alpha, dtype=_FLOATX) * negative_part
    return x

def res_block(inp):
    """ResNet B Block.
    TODO: decide whether bias is worth it
    TODO: Figure out whether moving_mean and moving_variance work.
    """
    kernel_shape = (1,)
    bias_shape = (1,)
    params_shape = (1,)

    weights = tf.get_variable('weights1', kernel_shape,
        initializer=tf.random_normal_initializer())
    beta = tf.get_variable('beta1', params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma1', params_shape, initializer=tf.ones_initializer)
    moving_mean = tf.get_variable('moving_mean1', params_shape, initializer=tf.zeros_initializer)
    moving_variance = tf.get_variable('moving_variance1', params_shape, initializer=tf.ones_initializer)

    h = tf.nn.conv2d(inp, weights, padding='SAME', strides=[1, 1, 1, 1])
    h = tf.nn.batch_normalization(h, moving_mean, moving_variance, beta, gamma, BN_EPSILON)
    h = tf.nn.relu(h)

    weights = tf.get_variable('weights2', kernel_shape,
        initializer=tf.random_normal_initializer())
    beta = tf.get_variable('beta2', params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma2', params_shape, initializer=tf.ones_initializer)
    moving_mean = tf.get_variable('moving_mean2', params_shape, initializer=tf.zeros_initializer)
    moving_variance = tf.get_variable('moving_variance2', params_shape, initializer=tf.ones_initializer)

    h = tf.nn.conv2d(inp, weights, padding='SAME', strides=[1, 1, 1, 1])
    h = tf.nn.batch_normalization(h, moving_mean, moving_variance, beta, gamma, BN_EPSILON)

    return inp + h

def deconv_block(inp):
    """
    TODO: Figure out what all these parameters do.
    TODO: Decide whether or not to add bias.
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    """
    k_h, k_w = 5, 5
    d_h, d_w = 2, 2
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=0.02))
    h = tf.nn.conv2d_transpose(inp, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    h = tf.nn.relu(h)

    return h

def conv_block(inp, relu=False, leaky_relu=False, bn=False):
    weights = tf.get_variable('weights', kernel_shape,
        initializer=tf.random_normal_initializer())
    h = tf.nn.conv2d(inp, weights, padding='SAME', strides=[1, 1, 1, 1])

    if relu:
        h = tf.nn.relu(h)

    if leaky_relu:
        h = relu(h, alpha=0.1)

    if bn:
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
        moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer)
        moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer)
        h = tf.nn.batch_normalization(h, moving_mean, moving_variance, beta, gamma, BN_EPSILON)

    return h

def dense_block(inp, leaky_relu=False, sigmoid=False):
    w = tf.get_variable("w", [shape[1], output_size], tf.random_normal_initializer(stddev=0.02))

    b = tf.get_variable("b", [output_size], tf.constant_initializer(0.0))

    h = tf.matmul(inp, w) + b

    if leaky_relu:
        h = relu(h, alpha=0.1)

    if sigmoid:
        h = tf.nn.sigmoid(h)

    return h

class GAN(object):
    def __init__(self):
        self.g_images = tf.placeholder(tf.float32, [self.batch_size, 96, 96, 3])
        self.d_images = tf.placeholder(tf.float32, [self.batch_size, 784, 784, 3])

    def build_model(self):
        with tf.variable_scope("G"):
            self.G = self.generator()

        with tf.variable_scope("D"):
            self.D1 = self.discriminator(self.G)
            self.D2 = self.discriminator(self.d_images)

    def generator(self):
        """Returns model generator, which is a DeConvNet.
        Assumed properties:
            gen_input - a scalar
            batch_size
            dimensions of filters and other hyperparameters.
            ...
        """
        with tf.variable_scope("conv1"):
            h = conv_block(self.g_images, relu=True)

        with tf.variable_scope("res1"):
            h = res_block(h)

        with tf.variable_scope("res2"):
            h = res_block(h)

        with tf.variable_scope("res3"):
            h = res_block(h)

        with tf.variable_scope("res4"):
            h = res_block(h)

        with tf.variable_scope("res5"):
            h = res_block(h)

        with tf.variable_scope("res6"):
            h = res_block(h)

        with tf.variable_scope("deconv1"):
            h = deconv_block(h)

        with tf.variable_scope("deconv2"):
            h = deconv_block(h)

        with tf.variable_scope("conv2"):
            h = conv_block(h)

        return h

    def discriminator(self, inp):
        """Returns model discriminator.
        Assumed properties:
            disc_input - an image tensor
            G - a generator
            ...
        """
        with tf.variable_scope("conv1"):
            h = conv_block(inp, leaky_relu=True)

        with tf.variable_scope("conv2"):
            h = conv_block(h, leaky_relu=True, bn=True)

        with tf.variable_scope("conv3"):
            h = res_block(h, leaky_relu=True, bn=True)

        with tf.variable_scope("conv4"):
            h = res_block(h, leaky_relu=True, bn=True)

        with tf.variable_scope("conv5"):
            h = res_block(h, leaky_relu=True, bn=True)

        with tf.variable_scope("conv6"):
            h = res_block(h, leaky_relu=True, bn=True)

        with tf.variable_scope("conv7"):
            h = res_block(h, leaky_relu=True, bn=True)

        with tf.variable_scope("conv8"):
            h = res_block(h, leaky_relu=True, bn=True)

        with tf.variable_scope("deconv2"):
            h = deconv_block(h)

        with tf.variable_scope("conv2"):
            h = conv_block(h)

        return h

class SuperRes(object):
    def __init__(self, sess, loader):
        self.sess = sess
        self.batch = loader.batch()

        self.GAN = GAN()
        d_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.GAN.d_loss)
        g_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.GAN.g_loss)

    def train_model(self):
        tf.initialize_all_variables().run()

        # Train
        for epoch in range(num_epochs):
            batch_xs, batch_ys = sess.run(self.batch)
            self.sess.run(self.d_optim, feed_dict={y: batch_ys})
            self.sess.run(self.g_optim, feed_dict={x: batch_xs})
            self.sess.run(self.g_optim, feed_dict={x: batch_xs})

    def test_model(self):
        pass

def main():
    sess = tf.Session()
    file_list_2 = glob.glob("/etc/gan_images/high_res/*.png")
    file_info_2 = (file_list_2, 768, 768)
    loader = Loader(file_info_2)
    model = SuperRes(sess, loader)
    model.train_model()

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
