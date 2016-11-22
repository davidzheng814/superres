import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import numpy as np
import glob
import argparse
import logging

from os.path import join

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# TODO Download all images
# TODO Split into training and validation sets
# TODO Add a pass through the validation set after each epoch
# TODO Add a summary writer for the validation set as well
# TODO Checkpoint save training weights
# TODO Start making hyperparameters command line options

IMAGES = "images/*.png"
LOGS_DIR = "logs/"

HR_HEIGHT = 384
HR_WIDTH = 384
r = 4
LR_HEIGHT = HR_HEIGHT / r
LR_WIDTH = HR_WIDTH / r
NUM_CHANNELS = 3
BATCH_SIZE = 50
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4
BN_EPSILON = 0.001
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
UPDATE_OPS_COLLECTION = 'update_ops'
BETA_1 = 0.9
RANDOM_SEED = 1337 

class Loader(object):
    def __init__(self, high_res_info):
        global NUM_IMAGES, NUM_BATCHES

        f2, self.h2, self.w2 = high_res_info
        self.q2 = tf.train.string_input_producer(f2)
        NUM_IMAGES = len(f2)
        NUM_BATCHES = len(f2) / BATCH_SIZE
        logging.info("Running on %d images" % (NUM_IMAGES,))

    def _get_pipeline(self, q, h, w):
        reader = tf.WholeFileReader()
        key, value = reader.read(q)
        raw_img = tf.image.decode_png(value, channels=NUM_CHANNELS)
        my_img = tf.image.per_image_whitening(raw_img)
        my_img = tf.random_crop(my_img, [HR_HEIGHT, HR_WIDTH, NUM_CHANNELS], seed=RANDOM_SEED)
        min_after_dequeue = 1
        capacity = min_after_dequeue + 3 * BATCH_SIZE
        batch = tf.train.shuffle_batch([my_img], batch_size=BATCH_SIZE, capacity=capacity,
                min_after_dequeue=min_after_dequeue, seed=RANDOM_SEED)
        small_batch = tf.image.resize_bicubic(batch, [LR_HEIGHT, LR_WIDTH])
        return small_batch, batch

    def batch(self):
        return self._get_pipeline(self.q2, self.h2, self.w2)

def relu_block(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def res_block(inp, is_training_cond):
    """ResNet B Block.
    TODO: decide whether bias is worth it
    """
    inp_shape = inp.get_shape()
    assert inp_shape[-1] == 64 # Must have 64 channels to start.

    with tf.variable_scope("resconv1"):
        h = conv_block(inp, relu=True, bn=True, is_training_cond=is_training_cond)

    with tf.variable_scope("resconv2"):
        h = conv_block(h, bn=True, is_training_cond=is_training_cond)

    return inp + h

def deconv_block(inp, relu=True, output_channels=64):
    """
    TODO: Decide whether or not to add bias.
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    """
    inp_shape = inp.get_shape()
    kernel_shape = (3, 3, output_channels, inp_shape[-1])
    output_shape = (int(inp_shape[0]), int(inp_shape[1] * 2), int(inp_shape[2] * 2), output_channels)
    strides = [1, 2, 2, 1]

    weights = tf.get_variable('weights', kernel_shape,
        initializer=tf.random_normal_initializer(stddev=0.02))
    h = tf.nn.conv2d_transpose(inp, weights, output_shape, strides)

    if relu:
        h = tf.nn.relu(h)

    return h

def conv_block(inp, relu=False, leaky_relu=False, bn=False,
               output_channels=64, stride=1, is_training_cond=None):
    inp_shape = inp.get_shape()
    kernel_shape = (3, 3, inp_shape[-1], output_channels)
    strides = [1, stride, stride, 1]

    weights = tf.get_variable('weights', kernel_shape,
        initializer=tf.random_normal_initializer())
    h = tf.nn.conv2d(inp, weights, strides, padding='SAME')

    if leaky_relu:
        h = relu_block(h, alpha=0.1)

    if bn:
        pass # TODO Figure out how to do BN
        # beta = tf.get_variable('beta', params_shape, 
        #     initializer=tf.zeros_initializer)
        # gamma = tf.get_variable('gamma', params_shape, 
        #     initializer=tf.ones_initializer)
        # moving_mean = tf.get_variable('moving_mean', params_shape, 
        #     initializer=tf.zeros_initializer, trainable=False)
        # moving_variance = tf.get_variable('moving_variance', params_shape, 
        #     initializer=tf.ones_initializer, trainable=False)
        # mean, variance = tf.nn.moments(h, [0, 1, 2])
        # update_moving_mean = moving_averages.assign_moving_average(moving_mean,
        #     mean, BN_DECAY)
        # update_moving_variance = moving_averages.assign_moving_average(moving_variance,
        #     variance, BN_DECAY)
        # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        # mean, variance = control_flow_ops.cond(is_training_cond, lambda: (mean, variance),
        #     lambda: (moving_mean, moving_variance))
        # h = tf.nn.batch_normalization(h, mean, variance, beta, gamma, BN_EPSILON)

    if relu:
        h = tf.nn.relu(h)

    return h

def dense_block(inp, leaky_relu=False, sigmoid=False,
                output_size=1024):
    inp_size = inp.get_shape()
    h = tf.reshape(inp, [int(inp_size[0]), -1])
    h_size = h.get_shape()[1]

    w = tf.get_variable("w", [h_size, output_size],
        initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [output_size],
        initializer=tf.constant_initializer(0.0))

    h = tf.matmul(h, w) + b

    if leaky_relu:
        h = relu_block(h, alpha=0.1)

    if sigmoid:
        h = tf.nn.sigmoid(h)

    return h

class GAN(object):
    def __init__(self):
        self.g_images = tf.placeholder(tf.float32, 
            [BATCH_SIZE, LR_HEIGHT, LR_WIDTH, NUM_CHANNELS])
        self.d_images = tf.placeholder(tf.float32,
            [BATCH_SIZE, HR_HEIGHT, HR_WIDTH, NUM_CHANNELS])
        self.is_training = tf.placeholder(tf.bool, [1])

    def build_model(self):
        with tf.variable_scope("G"):
            self.G = self.generator()

        with tf.variable_scope("D"):
            self.D = self.discriminator(self.d_images)
            tf.get_variable_scope().reuse_variables()
            self.DG = self.discriminator(self.G)

        # MSE Loss and Adversarial Loss for G
        self.mse_loss = tf.reduce_mean(tf.squared_difference(self.d_images, self.G))
        self.g_ad_loss = tf.reduce_mean(tf.neg(tf.log(self.DG)))

        self.g_loss = self.mse_loss + self.g_ad_loss
        tf.scalar_summary('g_loss', self.g_loss)

        # Real Loss and Adversarial Loss for D
        self.d_loss_real = tf.reduce_mean(tf.neg(tf.log(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.log(self.DG))

        self.d_loss = self.d_loss_real + self.d_loss_fake
        tf.scalar_summary('d_loss', self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'D/' in var.name]
        self.g_vars = [var for var in t_vars if 'G/' in var.name]

        # TODO Missing VGG loss and regularization loss. 
        # Also missing weighting on losses.

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
            h = res_block(h, self.is_training)

        with tf.variable_scope("res2"):
            h = res_block(h, self.is_training)

        with tf.variable_scope("res3"):
            h = res_block(h, self.is_training)

        with tf.variable_scope("res4"):
            h = res_block(h, self.is_training)

        with tf.variable_scope("res5"):
            h = res_block(h, self.is_training)

        with tf.variable_scope("res6"):
            h = res_block(h, self.is_training)

        with tf.variable_scope("deconv1"):
            h = deconv_block(h)

        with tf.variable_scope("deconv2"):
            h = deconv_block(h)

        with tf.variable_scope("conv2"):
            h = conv_block(h, output_channels=3)

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
            h = conv_block(h, leaky_relu=True, bn=True, 
                is_training_cond=self.is_training, stride=2)

        with tf.variable_scope("conv3"):
            h = conv_block(h, leaky_relu=True, bn=True, 
                is_training_cond=self.is_training, output_channels=128)

        with tf.variable_scope("conv4"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=128, stride=2)

        with tf.variable_scope("conv5"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=256, stride=1)

        with tf.variable_scope("conv6"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=256, stride=2)

        with tf.variable_scope("conv7"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=512, stride=1)

        with tf.variable_scope("conv8"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=512, stride=2)

        with tf.variable_scope("dense1"):
            h = dense_block(h, leaky_relu=True, output_size=1024)

        with tf.variable_scope("dense2"):
            h = dense_block(h, sigmoid=True, output_size=1)

        return h


class SuperRes(object):
    def __init__(self, sess, loader):
        logging.info("Building Model.")
        self.sess = sess
        self.loader = loader
        self.batch = loader.batch()

        self.GAN = GAN()
        self.GAN.build_model()

        self.g_mse_optim = (tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1)
            .minimize(self.GAN.mse_loss, var_list=self.GAN.g_vars))
        self.d_optim = (tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1)
            .minimize(self.GAN.d_loss, var_list=self.GAN.d_vars))
        self.g_optim = (tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1)
            .minimize(self.GAN.g_loss, var_list=self.GAN.g_vars))

    def train_model(self):
        merged = tf.merge_all_summaries()
        pre_train_writer = tf.train.SummaryWriter(join(LOGS_DIR, 'pretrain'), self.sess.graph)
        train_writer = tf.train.SummaryWriter(join(LOGS_DIR, 'train'), self.sess.graph)
        logging.info("Initializing Variables.")
        with self.sess as sess:
            tf.initialize_all_variables().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Pretrain
            logging.info("Begin Pre-Training")
            ind = 0
            for epoch in range(NUM_EPOCHS):
                logging.info("Pre-Training Epoch: %d" % (epoch,))
                for batch in range(NUM_BATCHES):
                    lr, hr = sess.run(self.batch)
                    summary, _ = self.sess.run([merged, self.g_mse_optim], feed_dict={
                        self.GAN.g_images: lr,
                        self.GAN.d_images: hr,
                        self.GAN.is_training: [True]
                    })
                    pre_train_writer.add_summary(summary, ind)

                    if ind % 10000 == 0:
                        logging.info("Pre-Training Iter: %d" % (ind,))

                    ind += 1

            logging.info("Begin Training")
            # Train
            ind = 0
            for epoch in range(NUM_EPOCHS):
                logging.info("Training Epoch: %d" % (epoch,))
                for batch in range(NUM_BATCHES):
                    lr, hr = sess.run(self.batch)
                    summary, _, __ = sess.run([merged, self.d_optim, self.g_optim], feed_dict={
                        self.GAN.g_images: lr,
                        self.GAN.d_images: hr,
                        self.GAN.is_training: [True]
                    })
                    train_writer.add_summary(summary, ind)

                    if ind % 10000 == 0:
                        logging.info("Pre-Training Iter: %d" % (ind,))

                    ind += 1

            coord.request_stop()
            coord.join(threads)

    def test_model(self):
        pass

def main():
    sess = tf.Session()
    file_list_2 = glob.glob(IMAGES)
    file_info_2 = (file_list_2, HR_HEIGHT, HR_WIDTH)
    loader = Loader(file_info_2)
    model = SuperRes(sess, loader)
    model.train_model()

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()
