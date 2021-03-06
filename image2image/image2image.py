#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Image2Image.py
# Author: Yuxin Wu

import cv2
import numpy as np
import tensorflow as tf
import glob
import os
import argparse
import shutil
import sys
from tensorpack import *
from tensorpack.utils.viz import stack_patches
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack import ModelDesc, SimpleTrainer

"""
To train Image-to-Image translation model with image pairs:
    ./Image2Image.py --data /path/to/datadir --mode {AtoB,BtoA}
    # datadir should contain jpg images of shpae 2s x s, formed by A and B
    # you can download some data from the original authors:
    # https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
Training visualization will appear be in tensorboard.
To visualize on test set:
    ./Image2Image.py --sample --data /path/to/test/datadir --mode {AtoB,BtoA} --load model
"""

BATCH = 1
IN_CH = 1
OUT_CH = 1
LAMBDA = 100
NF = 64  # number of filter


def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, alpha=0.2, name=name)


def visualize_tensors(name, imgs, scale_func=lambda x: (x + 1.) * 128., max_outputs=1):
    """Generate tensor for TensorBoard (casting, clipping)
    Args:
        name: name for visualization operation
        *imgs: multiple tensors as list
        scale_func: scale input tensors to fit range [0, 255]
    Example:
        visualize_tensors('viz1', [img1])
        visualize_tensors('viz2', [img1, img2, img3], max_outputs=max(30, BATCH))
    """
    xy = scale_func(tf.concat(imgs, axis=2))
    xy = tf.cast(tf.clip_by_value(xy, 0, 255), tf.uint8, name='viz')
    tf.summary.image(name, xy, max_outputs=30)


class Model(ModelDesc):
    def inputs(self):
        SHAPE = 128
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 2), 'input'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, OUT_CH), 'output')]

    def image2image(self, imgs):
        # imgs: input: 256x256xch
        # U-Net structure, it's slightly different from the original on the location of relu/lrelu

        with argscope(BatchNorm, training=True), \
                argscope(Dropout, is_training=True):
            with argscope(Conv2D, kernel_size=4, strides=2, activation=BNLReLU):
                e1 = Conv2D('conv1', imgs, NF, activation=tf.nn.leaky_relu)
                e2 = Conv2D('conv2', e1, NF * 2)
                e3 = Conv2D('conv3', e2, NF * 4)
                e4 = Conv2D('conv5', e3, NF * 8)
                e5 = Conv2D('conv6', e4, NF * 8)
                e6 = Conv2D('conv7', e5, NF * 8)
                e7 = Conv2D('conv8', e6, NF * 8, activation=BNReLU)  # 1x1
            with argscope(Conv2DTranspose, activation=BNReLU, kernel_size=4, strides=2):
                return (LinearWrap(e7)
                        .Conv2DTranspose('deconv1', NF * 8)
                        .Dropout()
                        .ConcatWith(e6, 3)
                        .Conv2DTranspose('deconv2', NF * 8)
                        .Dropout()
                        .ConcatWith(e5, 3)
                        .Conv2DTranspose('deconv3', NF * 8)
                        .Dropout()
                        .ConcatWith(e4, 3)
                        .Conv2DTranspose('deconv5', NF * 4)
                        .ConcatWith(e3, 3)
                        .Conv2DTranspose('deconv6', NF * 2)
                        .ConcatWith(e2, 3)
                        .Conv2DTranspose('deconv7', NF * 1)
                        .ConcatWith(e1, 3)
                        .Conv2DTranspose('deconv8', 2, activation=tf.identity)())  # TODO: clean

    def build_graph(self, input, output):

        input = input / 128.0 - 1
        output = tf.cast(output / 255.0, tf.int32)

        with argscope([Conv2D, Conv2DTranspose], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            fake_output = self.image2image(input)
        fake_output = tf.identity(fake_output, name='aaa')
        #TODO: clean
        errL1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output[:,:,:,0], logits=fake_output), name='L1_loss')

        score = tf.nn.softmax(fake_output)
        add_moving_summary(errL1)

        viz = tf.identity(tf.concat([input[:,:,:,:1], tf.cast(output, tf.float32), score[:,:,:,1:2]], axis=2), name='viz')

        if False:
            # tensorboard visualization
            if IN_CH == 1:
                input = tf.image.grayscale_to_rgb(input)
            if OUT_CH == 1:
                output = tf.image.grayscale_to_rgb(output)
                fake_output = tf.image.grayscale_to_rgb(fake_output)

            visualize_tensors('input,output,fake', [input, tf.cast(output, tf.float32), fake_output], max_outputs=max(30, BATCH))

        self.cost = errL1

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def split_input(img):
    """
    img: an RGB image of shape (s, 2s, 3).
    :return: [input, output]
    """
    # split the image into left + right pairs
    s = img.shape[0]
    assert img.shape[1] == 3 * s
    input, output, depth = img[:, :s, :], img[:, s:2*s, :], img[:, 2*s:3*s, :]
    input = np.concatenate([input, depth], axis=2)

    if args.mode == 'BtoA':
        assert False, 'Not supported feature'
        input, output = output, input

    if False:
        # TODO: Verify
        if IN_CH == 1:
            input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        if OUT_CH == 1:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    return [input, output]


def get_data(datadir):

    imgs = glob.glob(os.path.join(datadir, '*.png'))
    ds = ImageFromFile(imgs, channel=IN_CH, shuffle=True)

    ds = MapData(ds, lambda dp: split_input(dp[0]))
    augs = [imgaug.RandomCrop(128)]
    ds = AugmentImageComponents(ds, augs, (0, 1))
    ds = BatchData(ds, BATCH)
    ds = PrefetchData(ds, 100, 1)
    return ds


def sample(datadir, model_path, output_path):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input', 'output'],
        output_names=['viz'])

    imgs = glob.glob(os.path.join(datadir, '*.png'))
    ds = ImageFromFile(imgs, channel=IN_CH, shuffle=True)
    ds = MapData(ds, lambda dp: split_input(dp[0]))
    augs = [imgaug.CenterCrop(128)]
    ds = AugmentImageComponents(ds, augs, (0, 1))
    ds = BatchData(ds, BATCH)

    pred = SimpleDatasetPredictor(pred, ds)

    counter = 0
    for o in pred.get_result():
        for im in o[0]:
            np.save(os.path.join(output_path, f'{counter}.npy'), im)
            counter += 1
        # o = o[0][:, :, :, ::-1]
        # stack_patches(o, nr_row=3, nr_col=2, viz=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logger', help='logger_dir')
    parser.add_argument('--output_path', help='output_path for sampling')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='Image directory', required=True)
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    parser.add_argument('-b', '--batch', type=int, default=1)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch

    if args.sample:
        assert args.load
        assert args.output_path
        os.makedirs(args.output_path)
        sample(args.data, args.load, args.output_path)
    else:
        logger.set_logger_dir(args.logger, action='k')
        shutil.copyfile(sys.argv[0], os.path.join(args.logger, os.path.basename(sys.argv[0])))
        data = QueueInput(get_data(datadir = os.path.join(args.data, 'train')))

        config = AutoResumeTrainConfig(
            model=Model(),
            data=data,
            callbacks=[
                PeriodicTrigger(ModelSaver(), every_k_epochs=1),
                PeriodicTrigger(InferenceRunner(get_data(datadir=os.path.join(args.data, 'validation')), [ScalarStats('L1_loss')]), every_k_epochs=1),
                ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)])
            ],
            steps_per_epoch=data.size() // 8,
            max_epoch=300,
            session_init=SaverRestore(args.load) if args.load else None
        )



        launch_train_with_config(config, SimpleTrainer())
        # SimpleTrainer(data, Model()).train_with_defaults(
        #     callbacks=[
        #         PeriodicTrigger(ModelSaver(), every_k_epochs=3),
        #         ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)])
        #     ],
        #     steps_per_epoch=data.size(),
        #     max_epoch=300,
        #     session_init=SaverRestore(args.load) if args.load else None
        # )