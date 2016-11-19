import tensorflow as tf
import numpy as np
import argparse

class Loader(object):
    def __init__(self):
        pass

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
        self.loader = loader
        self.setup_model()

    def setup_model(self):
        self.GAN = GAN()
        d_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.GAN.d_loss)
        g_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.GAN.g_loss)

    def train_model(self):
        tf.initialize_all_variables().run()

        # Train
        for epoch in range(num_epochs):
            for idx, batch in self.loader.get_next_train_batch():
                batch_xs, batch_ys = self.loader.get_next_train_batch()
                self.sess.run(self.train_step, feed_dict={x: batch_xs, y_: batch_ys})

                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    def test_model(self):
        pass

def main():
    sess = tf.Session()
    loader = Loader()
    model = SuperRes(sess, loader)
    model.train_model()

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
