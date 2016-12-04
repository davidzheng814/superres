import tensorflow as tf
import numpy as np
import glob
import argparse
import logging
import os
import sys
import config as cfg
import re
from scipy.misc import imresize
from PIL import Image

from blocks import relu_block, res_block, deconv_block, conv_block, dense_block

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# TODO Download all images
# TODO Start making hyperparameters command line options
# TODO Check that things match with paper
# TODO Randomize train/test/validation sets (seed)

class Loader(object):
    def __init__(self, images):
        cfg.NUM_IMAGES = len(images)
        cfg.NUM_TRAIN_IMAGES = int(cfg.NUM_IMAGES * cfg.TRAIN_RATIO)
        cfg.NUM_VAL_IMAGES = int(cfg.NUM_IMAGES * cfg.VAL_RATIO)
        train_images = images[:cfg.NUM_TRAIN_IMAGES]
        val_images = images[cfg.NUM_TRAIN_IMAGES:cfg.NUM_TRAIN_IMAGES + cfg.NUM_VAL_IMAGES]
        test_images = images[cfg.NUM_TRAIN_IMAGES + cfg.NUM_VAL_IMAGES:]
        self.q_train = tf.train.string_input_producer(train_images)
        self.q_val = tf.train.string_input_producer(val_images)
        self.q_test = tf.train.string_input_producer(test_images)
        cfg.NUM_TRAIN_BATCHES = len(train_images) // cfg.BATCH_SIZE
        cfg.NUM_VAL_BATCHES = len(val_images) // cfg.BATCH_SIZE
        cfg.NUM_TEST_BATCHES = len(test_images) // cfg.BATCH_SIZE

    def _get_pipeline(self, q):
        reader = tf.WholeFileReader()
        key, value = reader.read(q)
        raw_img = tf.image.decode_jpeg(value, channels=cfg.NUM_CHANNELS)
        my_img = tf.random_crop(raw_img, [cfg.HR_HEIGHT, cfg.HR_WIDTH, cfg.NUM_CHANNELS],
                seed=cfg.RANDOM_SEED)
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * cfg.BATCH_SIZE
        batch = tf.train.shuffle_batch([my_img], batch_size=cfg.BATCH_SIZE, capacity=capacity,
                min_after_dequeue=min_after_dequeue, seed=cfg.RANDOM_SEED)
        small_batch = tf.image.resize_bicubic(batch, [cfg.LR_HEIGHT, cfg.LR_WIDTH])
        return (small_batch, batch)

    def batch(self):
        return (self._get_pipeline(self.q_train),
                self._get_pipeline(self.q_val),
                self._get_pipeline(self.q_test))

class GAN(object):
    def __init__(self):
        self.g_images = tf.placeholder(tf.float32, 
            [cfg.BATCH_SIZE, cfg.LR_HEIGHT, cfg.LR_WIDTH, cfg.NUM_CHANNELS])
        self.d_images = tf.placeholder(tf.float32,
            [cfg.BATCH_SIZE, cfg.HR_HEIGHT, cfg.HR_WIDTH, cfg.NUM_CHANNELS])
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.image_tensor = None

    def build_model(self, input_image=None):
        if input_image is not None:
            with tf.variable_scope("G", reuse=True) as scope:
                self.image_tensor = tf.placeholder(tf.float32, input_image.shape)
                scope.reuse_variables()
                self.G = self.generator()
        else:
            with tf.variable_scope("G"):
                self.G = self.generator()

            with tf.variable_scope("D") as scope:
                self.D = self.discriminator(self.d_images)
                scope.reuse_variables()
                self.DG = self.discriminator(self.G)

            # MSE Loss and Adversarial Loss for G
            self.mse_loss = tf.reduce_mean(
                    tf.squared_difference(self.d_images, self.G))
            self.g_ad_loss = (tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                    self.DG, tf.ones_like(self.DG))))

            self.g_loss = self.mse_loss + cfg.AD_LOSS_WEIGHT * self.g_ad_loss
            tf.scalar_summary('g_loss', self.g_loss)

            # Real Loss and Adversarial Loss for D
            self.d_loss_real = (tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                    self.D, tf.ones_like(self.D))))
            self.d_loss_fake = (tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                    self.DG, tf.zeros_like(self.DG))))

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
            if self.image_tensor != None:
                h = conv_block(self.image_tensor, relu=True)
            else:
                noise = tf.random_normal(self.g_images.get_shape(), stddev=.03 * 255)
                h = self.g_images + noise
                h = conv_block(self.g_images, relu=True)

        for i in range(1, 17):
            with tf.variable_scope("res" + str(i)):
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
            h = dense_block(h, output_size=1)

        return h


class SuperRes(object):
    def __init__(self, sess, loader):
        logging.info("Building Model.")
        self.sess = sess
        self.loader = loader
        self.train_batch, self.val_batch, self.test_batch = loader.batch()

        self.GAN = GAN()
        self.GAN.build_model()

        self.g_mse_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.mse_loss, var_list=self.GAN.g_vars))
        self.d_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.d_loss, var_list=self.GAN.d_vars))
        self.g_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.g_loss, var_list=self.GAN.g_vars))

        batchnorm_updates = tf.get_collection(cfg.UPDATE_OPS_COLLECTION)
        self.pretrain = tf.group(self.g_mse_optim, *batchnorm_updates)
        self.train = tf.group(self.d_optim, self.g_optim, *batchnorm_updates)

    def predict(self, input_name, output_name, init_vars):
        if init_vars == True:
            self._load_latest_checkpoint_or_initialize(tf.train.Saver())
        with Image.open(input_name) as image:
            image = np.asarray(image, dtype=np.uint8)
            image = imresize(image, 100 // cfg.r)
            image = np.reshape(image, (1,) + image.shape)
            test_GAN = GAN()
            test_GAN.build_model(image)
            generated_image = self.sess.run(
                [test_GAN.G],
                feed_dict={
                    test_GAN.image_tensor: image,
                    test_GAN.is_training: False
            })
            generated_image = np.uint8(generated_image[0][0])
            generated_image = Image.fromarray(generated_image).convert('RGB')
            generated_image.save(output_name)

    def _load_latest_checkpoint_or_initialize(self, saver, attempt_load=True):
        ckpt_files = list(filter(lambda x: "meta" not in x, glob.glob(cfg.CHECKPOINT + "*")))
        if attempt_load and len(ckpt_files) > 0:
            ckpt_files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)])
            logging.info("Loading params from " + ckpt_files[-1])
            saver.restore(self.sess, ckpt_files[-1])
            return ckpt_files[-1]
        else:
            logging.info("Initializing parameters")
            self.sess.run(tf.initialize_all_variables())
            return ""

    def _pretrain(self):
        lr, hr = self.sess.run(self.train_batch)
        summary, _, loss = self.sess.run(
            [self.merged, self.pretrain, self.GAN.mse_loss],
            feed_dict={
                self.GAN.g_images: lr,
                self.GAN.d_images: hr,
                self.GAN.is_training: True
        })
        return summary, loss

    def _train(self):
        """
        Returns (summary, mse_loss, g_ad_loss, g_loss, d_loss_real, d_loss_fake, d_loss)
        """
        lr, hr = self.sess.run(self.val_batch)
        res = self.sess.run(
            [self.train, self.merged,
             self.GAN.g_loss, self.GAN.mse_loss, self.GAN.g_ad_loss,
             self.GAN.d_loss, self.GAN.d_loss_real, self.GAN.d_loss_fake],
            feed_dict={
                self.GAN.g_images: lr,
                self.GAN.d_images: hr,
                self.GAN.is_training: True
        })

        return res[1:]

    def _val(self):
        """
        Returns (summary, mse_loss, g_ad_loss, g_loss, d_loss_real, d_loss_fake, d_loss)
        """
        lr, hr = self.sess.run(self.train_batch)
        res = self.sess.run(
            [self.merged,
             self.GAN.g_loss, self.GAN.mse_loss, self.GAN.g_ad_loss,
             self.GAN.d_loss, self.GAN.d_loss_real, self.GAN.d_loss_fake],
            feed_dict={
                self.GAN.g_images: lr,
                self.GAN.d_images: hr,
                self.GAN.is_training: False
        })

        return res

    def _test(self):
        """
        Returns (summary, mse_loss, g_ad_loss, g_loss, d_loss_real, d_loss_fake, d_loss)
        """
        lr, hr = self.sess.run(self.test_batch)
        res = self.sess.run(
            [self.merged,
             self.GAN.g_loss, self.GAN.mse_loss, self.GAN.g_ad_loss,
             self.GAN.d_loss, self.GAN.d_loss_real, self.GAN.d_loss_fake],
            feed_dict={
                self.GAN.g_images: lr,
                self.GAN.d_images: hr,
                self.GAN.is_training: False
        })

        return res

    def _print_losses(self, losses, count):
        avg_losses = [x / count for x in losses]
        logging.info("G Loss: %f, MSE Loss: %f, Ad Loss: %f"
                % (avg_losses[0], avg_losses[1], avg_losses[2]))
        logging.info("D Loss: %f, Real Loss: %f, Fake Loss: %f"
                % (avg_losses[3], avg_losses[4], avg_losses[5]))

    def train_model(self):
        logging.info("Running on %d images" % (cfg.NUM_IMAGES,))
        self.merged = tf.merge_all_summaries()
        self.pre_train_writer = tf.train.SummaryWriter(os.path.join(cfg.LOGS_DIR, 'pretrain'),
                self.sess.graph)
        self.train_writer = tf.train.SummaryWriter(os.path.join(cfg.LOGS_DIR, 'train'),
                self.sess.graph)
        self.val_writer = tf.train.SummaryWriter(os.path.join(cfg.LOGS_DIR, 'val'),
                self.sess.graph)
        saver = tf.train.Saver(max_to_keep=None)
        ckpt = self._load_latest_checkpoint_or_initialize(saver, attempt_load=cfg.USE_CHECKPOINT)
        match = re.search(r'\d+$', ckpt)
        done_batch = int(match.group(0)) if match else 0

        sess = self.sess
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if "adversarial" not in ckpt:
            # Pretrain
            logging.info("Begin Pre-Training")
            ind = 0
            for epoch in range(done_batch + 1, cfg.NUM_PRETRAIN_EPOCHS + 1):
                logging.info("Pre-Training Epoch: %d" % (epoch,))
                loss_sum = 0
                for batch in range(cfg.NUM_TRAIN_BATCHES):
                    summary, loss = self._pretrain()
                    self.pre_train_writer.add_summary(summary, ind)
                    loss_sum += loss
                    ind += 1
                logging.info("Epoch MSE Loss: %f" % (loss_sum / cfg.NUM_TRAIN_BATCHES,))

                if epoch%10==0:
                    logging.info("Saving Checkpoint")
                    saver.save(sess, cfg.CHECKPOINT + str(epoch))
            done_batch = 0
        else:
            logging.info("Skipping Pre-Training")

        logging.info("Begin Training")
        # Adversarial training
        ind = 0
        for epoch in range(done_batch + 1, cfg.NUM_TRAIN_EPOCHS + 1):
            logging.info("Training Epoch: %d" % (epoch,))
            losses = [0 for _ in range(6)]
            for batch in range(cfg.NUM_TRAIN_BATCHES):
                res = self._train()
                self.train_writer.add_summary(res[0], ind)
                losses = [x + y for x, y in zip(losses, res[1:])]
                ind += 1

            logging.info("Epoch Training Losses")
            self._print_losses(losses, cfg.NUM_TRAIN_BATCHES)

            # Validation
            losses = [0 for _ in range(6)]
            for batch in range(cfg.NUM_VAL_BATCHES):
                res = self._val()
                self.val_writer.add_summary(res[0], ind)
                losses = [x + y for x, y in zip(losses, res[1:])]
                ind += 1

            logging.info("Epoch Validation Losses")
            self._print_losses(losses, cfg.NUM_VAL_BATCHES)

            if epoch%5==0:
                logging.info("Saving Checkpoint (Adversarial)")
                saver.save(sess, cfg.CHECKPOINT + "_adversarial" + str(epoch))

        coord.request_stop()
        coord.join(threads)

    def test_model(self):
        val_writer = tf.train.SummaryWriter(join(cfg.LOGS_DIR, 'test'), self.sess.graph)

        with self.sess as sess:
            logging.info("Begin Testing")
            # Test
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            ind = 0
            for batch in range(cfg.NUM_TEST_BATCHES):
                lr, hr = sess.run(self.test_batch)
                res = self._test()
                test_writer.add_summary(res[0], ind)
                losses = [x + y for x, y in zip(losses, res[1:])]
                ind += 1

            logging.info("Test Losses")
            self._print_losses(losses, cfg.NUM_TEST_BATCHES)

            coord.request_stop()
            coord.join(threads)

def main():
    sess = tf.Session()
    file_list = glob.glob(cfg.IMAGES)
    loader = Loader(file_list)
    model = SuperRes(sess, loader)
    model.train_model()
    model.predict("/home/images/imagenet/n00007846_80134.JPEG", "output_image.JPEG", init_vars=False)
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()
