import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import glob
import argparse
import logging
import os
import sys
import config as cfg
import re
from scipy.misc import imresize, toimage
from PIL import Image
from scipy import signal, ndimage

from blocks import relu_block, res_block, deconv_block, conv_block, dense_block

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# TODO Download all images
# TODO Start making hyperparameters command line options
# TODO Check that things match with paper
# TODO Randomize train/test/validation sets (seed)

class Loader(object):
    def __init__(self, train_images, val_images, test_images):
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
        self.test_images = None

    def build_model(self, use_fft=False, input_images=None):
        if input_images is not None:
            with tf.variable_scope("G", reuse=True) as scope:
                self.test_images = tf.placeholder(tf.float32, input_images.shape)
                scope.reuse_variables()
                self.G = self.generator(reuse=True)
        else:
            with tf.variable_scope("G"):
                self.G = self.generator(reuse=False)

            with tf.variable_scope("D") as scope:
                if use_fft:
                    self.D = self.fft_discriminator(self.d_images)
                    scope.reuse_variables()
                    self.DG = self.fft_discriminator(self.G)
                else:
                    self.D = self.discriminator(self.d_images, reuse=False)
                    scope.reuse_variables()
                    self.DG = self.discriminator(self.G, reuse=True)

            # MSE Loss and Adversarial Loss for G
            self.mse_loss = tf.reduce_mean(
                tf.squared_difference(self.d_images, self.G))
            # self.mse_loss = tf.reduce_mean(tf.abs(self.d_images - self.G))
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

    def generator(self, reuse=False):
        """Returns model generator, which is a DeConvNet.
        Assumed properties:
            gen_input - a scalar
            batch_size
            dimensions of filters and other hyperparameters.
            ...
        """
        with tf.variable_scope("conv1"):
            if self.test_images is not None:
                h = conv_block(self.test_images, relu=True, reuse=reuse)
            else:
                # noise = tf.random_normal(self.g_images.get_shape(), stddev=.03 * 255)
                h = self.g_images
                h = conv_block(self.g_images, relu=True, reuse=reuse)

        for i in range(1, 16):
            with tf.variable_scope("res" + str(i)):
                h = res_block(h, self.is_training, reuse=reuse)

        with tf.variable_scope("deconv1"):
            h = deconv_block(h)

        with tf.variable_scope("conv2"):
            h = conv_block(h, output_channels=3, reuse=reuse)

        return h

    def fft_discriminator(self, inp):
        shuffled_inp = tf.transpose(inp, perm=[0, 3, 1, 2])
        inp_fft = tf.fft2d(tf.cast(shuffled_inp, tf.complex64))
        amp = tf.complex_abs(inp_fft)
        with tf.variable_scope("dense1"):
            h = dense_block(amp, leaky_relu=True, output_size=1024)
        with tf.variable_scope("dense2"):
            h = dense_block(h, output_size=1)
        return h

    def discriminator(self, inp, reuse=False):
        """Returns model discriminator.
        Assumed properties:
            disc_input - an image tensor
            G - a generator
            ...
        """
        with tf.variable_scope("conv1"):
            h = conv_block(inp, leaky_relu=True, reuse=reuse)

        with tf.variable_scope("conv2"):
            h = conv_block(h, leaky_relu=True, bn=True, 
                is_training_cond=self.is_training, stride=2, reuse=reuse)

        with tf.variable_scope("conv3"):
            h = conv_block(h, leaky_relu=True, bn=True, 
                is_training_cond=self.is_training, output_channels=128,
                reuse=reuse)

        with tf.variable_scope("conv4"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=128, stride=2,
                reuse=reuse)

        with tf.variable_scope("conv5"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=256, stride=1,
                reuse=reuse)

        with tf.variable_scope("conv6"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=256, stride=2,
                reuse=reuse)

        with tf.variable_scope("conv7"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=512, stride=1,
                reuse=reuse)

        with tf.variable_scope("conv8"):
            h = conv_block(h, leaky_relu=True, bn=True,
                is_training_cond=self.is_training, output_channels=512, stride=2,
                reuse=reuse)

        with tf.variable_scope("dense1"):
            h = dense_block(h, leaky_relu=True, output_size=1024)

        with tf.variable_scope("dense2"):
            h = dense_block(h, output_size=1)

        return h

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def to_y(img):
    return .229 * img[:,:,0] + .587 * img[:,:,1] + .114 * img[:,:,2]

def ssim(img1, img2, cs_map=False):
    img1 = to_y(img1.astype(np.uint8)[4:-4,4:-4])
    img2 = to_y(img2.astype(np.uint8)[4:-4,4:-4])
    size = (11, 11)
    sigma = 1.5
    window = matlab_style_gauss2D(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
            (sigma1_sq + sigma2_sq + C2))).mean()

class SuperRes(object):
    def __init__(self, sess, loader):
        logging.info("Building Model.")
        self.sess = sess
        self.loader = loader
        self.train_batch, self.val_batch, self.test_batch = loader.batch()

        self.GAN = GAN()
        self.GAN.build_model(use_fft=False)

        self.g_mse_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.mse_loss, var_list=self.GAN.g_vars))
        self.d_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.d_loss, var_list=self.GAN.d_vars))
        self.g_optim = (tf.train.AdamOptimizer(cfg.LEARNING_RATE, beta1=cfg.BETA_1)
            .minimize(self.GAN.g_loss, var_list=self.GAN.g_vars))

        batchnorm_updates = tf.get_collection(ops.GraphKeys.UPDATE_OPS)
        self.pretrain = tf.group(self.g_mse_optim, *batchnorm_updates)
        self.train = tf.group(self.d_optim, self.g_optim, *batchnorm_updates)

    def predict(self, input_name, output_name, init_vars=False):
        if init_vars == True:
            self._load_latest_checkpoint_or_initialize(tf.train.Saver())
        with Image.open(input_name) as image:
            hr = np.asarray(image, dtype=np.uint8)
            w = hr.shape[0] - hr.shape[0] % 4
            h = hr.shape[1] - hr.shape[1] % 4
            hr = hr[:w,:h]
            lr = imresize(hr, 100 // cfg.r, interp='bicubic')
            bicubic = imresize(lr, cfg.r * 100, interp='bicubic')
            image = np.reshape(lr, (1,) + lr.shape)
            test_GAN = GAN()
            test_GAN.build_model(input_images=image)
            sr = self.sess.run(
                [test_GAN.G],
                feed_dict={
                    test_GAN.test_images: image,
                    test_GAN.is_training: False
            })
            sr = np.maximum(np.minimum(sr[0][0], 255.0), 0.0)

            logging.info("SSIM - Bicubic %f, SR %f", ssim(bicubic, hr), ssim(sr, hr))

            toimage(lr, cmin=0., cmax=255.).save(output_name + '_lr.JPEG')
            toimage(bicubic, cmin=0., cmax=255.).save(output_name + '_bc.JPEG')
            toimage(hr, cmin=0., cmax=255.).save(output_name + '_hr.JPEG')
            toimage(sr, cmin=0., cmax=255.).save(output_name + '_sr.JPEG')

    def predict_all_2x(self, file_list, output_directory, scale, init_vars=False):
        if init_vars == True:
            self._load_latest_checkpoint_or_initialize(tf.train.Saver())
        batch_size = 50
        test_GAN = GAN()
        test_GAN.build_model(input_images=np.zeros((batch_size, int(cfg.CROP_HEIGHT/scale), int(cfg.CROP_WIDTH/scale), 3)))
        for i, image_file in enumerate(file_list):
            if not i % batch_size:
                images = []
            with Image.open(image_file) as image:
                lr = np.asarray(image, dtype=np.uint8)
                image_name = image_file.split("/")[-1]
                images.append(lr)
            if not (i + 1) % batch_size:
                images = np.array(images)
                sr = self.sess.run(
                    [test_GAN.G],
                    feed_dict={
                        test_GAN.test_images: images,
                        test_GAN.is_training: False
                })
                sr = sr[0]
                for j, image in enumerate(sr):
                    image = np.maximum(np.minimum(image, 255.0), 0.0)
                    toimage(image, cmin=0., cmax=255.).save(output_directory + file_list[j].split("/")[-1])

    def _load_latest_checkpoint_or_initialize(self, saver, attempt_load=True):
        if cfg.WEIGHTS:
            logging.info("Loading params from " + cfg.WEIGHTS)
            saver.restore(self.sess, cfg.WEIGHTS)
            return cfg.WEIGHTS
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

                if epoch % 10 == 0:
                    logging.info("Saving Checkpoint")
                    saver.save(sess, cfg.CHECKPOINT + str(epoch))
            done_batch = 0
        else:
            logging.info("Skipping Pre-Training")

        logging.info("Begin Training")
        # Adversarial training
        ind = 0
        if not cfg.PRETRAIN_ONLY:
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

                if epoch % 5 == 0:
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

def save_truth_images(file_list, output_directory):
    for image_file in file_list:
        with Image.open(image_file) as image:
            hr = np.asarray(image, dtype=np.uint8)
            x, y = hr.shape[0:2]
            if x >= cfg.CROP_HEIGHT and y >= cfg.CROP_WIDTH:
                image_name = image_file.split("/")[-1]
                hr = hr[0:cfg.CROP_HEIGHT, 0:cfg.CROP_WIDTH]
                hr_image = Image.fromarray(hr)
                hr_image.save(output_directory + "/truth4x/" + image_name)
                    
                lr2x = imresize(hr, 100 // 4, interp='bicubic')
                lr2x_image = Image.fromarray(lr2x)
                lr2x_image.save(output_directory + "/truth2x/" + image_name)
                    
                lr4x = imresize(lr2x, 100 // 2, interp='bicubic')
                lr4x_image = Image.fromarray(lr4x)
                lr4x_image.save(output_directory + "/truth/" + image_name)
 
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    file_list = glob.glob(cfg.IMAGES)
    if cfg.MAX_FILES:
        file_list = file_list[:cfg.MAX_FILES]
    cfg.NUM_IMAGES = len(file_list)
    cfg.NUM_TRAIN_IMAGES = int(cfg.NUM_IMAGES * cfg.TRAIN_RATIO)
    cfg.NUM_VAL_IMAGES = int(cfg.NUM_IMAGES * cfg.VAL_RATIO)
    train_images = file_list[:cfg.NUM_TRAIN_IMAGES]
    val_images = file_list[cfg.NUM_TRAIN_IMAGES:cfg.NUM_TRAIN_IMAGES + cfg.NUM_VAL_IMAGES]
    test_images = file_list[cfg.NUM_TRAIN_IMAGES + cfg.NUM_VAL_IMAGES:]

    truth_list = glob.glob(cfg.OUTPUT_DIR + "/truth/*")
    truth2x_list = glob.glob(cfg.OUTPUT_DIR + "/truth2x/*")
    truth4x_list = glob.glob(cfg.OUTPUT_DIR + "/truth4x/*")
    sr2x_list = glob.glob(cfg.OUTPUT_DIR + "/sr2x/*")

    loader = Loader(train_images, val_images, test_images)
    model = SuperRes(sess, loader)

    TEST_IMGS = [
        "/home/images/imagenet/n09287968_7641.JPEG",
        "/home/images/imagenet/n00523513_12670.JPEG",
        "/home/images/imagenet/n00007846_80134.JPEG"
    ]
    OUT_FILE = "images/test_{i}"
    
    if cfg.SAVE_TRUTH:
        save_truth_images(file_list, cfg.OUTPUT_DIR)
    elif cfg.PREDICT_4X:
        model.predict_all_2x(sr2x_list, cfg.OUTPUT_DIR + "/sr4x/", 2, init_vars=True)
    elif cfg.PREDICT_2X:
        model.predict_all_2x(truth_list, cfg.OUTPUT_DIR + "/sr2x/", 4, init_vars=True)
    elif cfg.PREDICT_ONLY:
        for i, img in enumerate(TEST_IMGS):
            out_file = OUT_FILE.replace("{i}", str(i))
            model.predict(img, out_file, init_vars=True)
    else:
        model.train_model()
        for i, img in enumerate(TEST_IMGS):
            out_file = OUT_FILE.replace("{i}", str(i))
            model.predict(img, out_file, init_vars=False)
    sess.close()
                  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--mem', type=float)
    parser.add_argument('--use-ckpt', action="store_true")
    parser.add_argument('--no-ckpt', action="store_true")
    parser.add_argument('--save-truth', action="store_true")
    parser.add_argument('--pretrain-only', action="store_true")
    parser.add_argument('--predict-only', action="store_true")
    parser.add_argument('--predict-4x', action="store_true")
    parser.add_argument('--predict-2x', action="store_true")
    parser.add_argument('--weights', type=str)
    parser.add_argument('--max-files', type=int)

    args = parser.parse_args()
    if args.num_epochs:
        cfg.NUM_EPOCHS = args.num_epochs
    if args.batch_size:
        cfg.NUM_EPOCHS = args.batch_size
    if args.use_ckpt:
        cfg.USE_CHECKPOINT = True
    if args.no_ckpt:
        cfg.USE_CHECKPOINT = False
    if args.max_files:
        cfg.MAX_FILES = args.max_files
    if args.save_truth:
        cfg.SAVE_TRUTH = True
    if args.pretrain_only:
        cfg.PRETRAIN_ONLY = True
    if args.predict_only:
        cfg.PREDICT_ONLY = True
    if args.predict_4x:
        cfg.PREDICT_4X = True
    if args.predict_2x:
        cfg.PREDICT_2X = True
    if args.mem:
        cfg.MEM_FRAC = args.mem
    if args.weights:
        cfg.USE_CHECKPOINT = True
        cfg.WEIGHTS = args.weights

    main()
