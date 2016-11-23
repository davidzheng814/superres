import tensorflow as tf
import glob
import argparse
import logging
import os

from blocks import relu_block, res_block, deconv_block, conv_block, dense_block

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# TODO Download all images
# TODO Split into training and validation sets
# TODO Add a pass through the validation set after each epoch
# TODO Add a summary writer for the validation set as well
# TODO Start making hyperparameters command line options

IMAGES = "images/*.png"
LOGS_DIR = "logs/"
CHECKPOINT = "checkpoint/weights.ckpt"

HR_HEIGHT = 384
HR_WIDTH = 384
r = 4
LR_HEIGHT = HR_HEIGHT // r
LR_WIDTH = HR_WIDTH // r
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
        NUM_BATCHES = len(f2) // BATCH_SIZE
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
        pre_train_writer = tf.train.SummaryWriter(os.path.join(LOGS_DIR, 'pretrain'), self.sess.graph)
        train_writer = tf.train.SummaryWriter(os.path.join(LOGS_DIR, 'train'), self.sess.graph)
        saver = tf.train.Saver()
        with self.sess as sess:
            if os.path.isfile(CHECKPOINT):
                logging.info("Restoring saved parameters")
                saver.restore(sess, CHECKPOINT)
            else:
                logging.info("Initializing parameters")
                sess.run(tf.initialize_all_variables())
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

                    if ind % 1000 == 0:
                        saver.save(sess, CHECKPOINT)
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

                    if ind % 1000 == 0:
                        saver.save(sess, CHECKPOINT)
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
