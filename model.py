import tensorflow as tf
import numpy as np
import glob
import argparse

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

class GAN(object):
    def __init__(self):
        pass

    def build_model():
        if self.y_dim:
            self.y = tf.placeholder(tf.float32)

    def generator():
        """Returns model generator, which is a DeConvNet.
        Assumed properties:
            gen_input - a scalar
            batch_size
            dimensions of filters and other hyperparameters.
            ...
        """
        h1 = tf.reshape(self.input, [self.batch_size, 1, 1, self.y_dim])

        for i in range(3):
            tf.nn.conv2d()

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def discriminator():
        """Returns model discriminator.
        Assumed properties:
            disc_input - an image tensor
            G - a generator
            ...
        """

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
