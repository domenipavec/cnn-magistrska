#!/usr/bin/env python3
import os
import time

import numpy as np
import cv2
import tensorflow as tf

from darkflow.net.build import TFNet

images = '../sample_img/'
images = '/home/domen/smb/'
use_camera = False


def _iterate_files():
    while 1:
        for subdir, dirs, files in os.walk(images):
            for file in files:
                yield os.path.join(subdir, file)


tfnet = TFNet({"model": "tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights"})

with tfnet.graph.as_default():
    layers = tfnet.darknet.layers

    def _get_var(layer, name):
        return tf.get_variable(
            'layer_%d_%s' % (layer, name),
            shape=layers[layer].wshape[name],
            initializer=layers[layer].w[name],
        )

    inpt = tf.placeholder(tf.float32, shape=(1, 416, 416, 3))

    conv1 = tf.nn.conv2d(
        inpt, layers[0].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch1 = tf.nn.batch_normalization(
        conv1,
        _get_var(0, 'moving_mean'),
        _get_var(0, 'moving_variance'),
        offset=None,
        scale=_get_var(0, 'gamma'),
        variance_epsilon=1e-5,
    )

    biased1 = tf.nn.bias_add(batch1, layers[0].w['biases'])

    leaky1 = tf.nn.leaky_relu(biased1, alpha=.1)

    maxpool1 = tf.nn.max_pool(
        leaky1,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
    )

    conv2 = tf.nn.conv2d(
        maxpool1, layers[3].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch2 = tf.nn.batch_normalization(
        conv2,
        _get_var(3, 'moving_mean'),
        _get_var(3, 'moving_variance'),
        offset=None,
        scale=_get_var(3, 'gamma'),
        variance_epsilon=1e-5,
    )

    biased2 = tf.nn.bias_add(batch2, layers[3].w['biases'])

    leaky2 = tf.nn.leaky_relu(biased2, alpha=.1)

    maxpool2 = tf.nn.max_pool(
        leaky2,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
    )

    conv3 = tf.nn.conv2d(
        maxpool2, layers[6].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch3 = tf.nn.batch_normalization(
        conv3,
        _get_var(6, 'moving_mean'),
        _get_var(6, 'moving_variance'),
        offset=None,
        scale=_get_var(6, 'gamma'),
        variance_epsilon=1e-5,
    )

    biased3 = tf.nn.bias_add(batch3, layers[6].w['biases'])

    leaky3 = tf.nn.leaky_relu(biased3, alpha=.1)

    maxpool3 = tf.nn.max_pool(
        leaky3,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
    )

    conv4 = tf.nn.conv2d(
        maxpool3, layers[9].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch4 = tf.nn.batch_normalization(
        conv4,
        _get_var(9, 'moving_mean'),
        _get_var(9, 'moving_variance'),
        offset=None,
        scale=_get_var(9, 'gamma'),
        variance_epsilon=1e-5,
    )

    biased4 = tf.nn.bias_add(batch4, layers[9].w['biases'])

    leaky4 = tf.nn.leaky_relu(biased4, alpha=.1)

    maxpool4 = tf.nn.max_pool(
        leaky4,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
    )

    conv5 = tf.nn.conv2d(
        maxpool4, layers[12].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch5 = tf.nn.batch_normalization(
        conv5,
        _get_var(12, 'moving_mean'),
        _get_var(12, 'moving_variance'),
        offset=None,
        scale=_get_var(12, 'gamma'),
        variance_epsilon=1e-5,
    )

    biased5 = tf.nn.bias_add(batch5, layers[12].w['biases'])

    leaky5 = tf.nn.leaky_relu(biased5, alpha=.1)

    maxpool5 = tf.nn.max_pool(
        leaky5,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
    )

    conv6 = tf.nn.conv2d(
        maxpool5, layers[15].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch6 = tf.nn.batch_normalization(
        conv6,
        _get_var(15, 'moving_mean'),
        _get_var(15, 'moving_variance'),
        offset=None,
        scale=_get_var(15, 'gamma'),
        variance_epsilon=1e-5,
    )

    biased6 = tf.nn.bias_add(batch6, layers[15].w['biases'])

    leaky6 = tf.nn.leaky_relu(biased6, alpha=.1)

    maxpool6 = tf.nn.max_pool(
        leaky6,
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 1, 1],
        padding='SAME',
    )

    conv7 = tf.nn.conv2d(
        maxpool6, layers[18].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch7 = tf.nn.batch_normalization(
        conv7,
        _get_var(18, 'moving_mean'),
        _get_var(18, 'moving_variance'),
        offset=None,
        scale=_get_var(18, 'gamma'),
        variance_epsilon=1e-5,
    )

    biased7 = tf.nn.bias_add(batch7, layers[18].w['biases'])

    leaky7 = tf.nn.leaky_relu(biased7, alpha=.1)

    conv8 = tf.nn.conv2d(
        leaky7, layers[20].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch8 = tf.nn.batch_normalization(
        conv8,
        _get_var(20, 'moving_mean'),
        _get_var(20, 'moving_variance'),
        offset=None,
        scale=_get_var(20, 'gamma'),
        variance_epsilon=1e-5,
    )

    biased8 = tf.nn.bias_add(batch8, layers[20].w['biases'])

    leaky8 = tf.nn.leaky_relu(biased8, alpha=.1)

    conv9 = tf.nn.conv2d(
        leaky8, layers[22].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    biased9 = tf.nn.bias_add(conv9, layers[22].w['biases'])

    output = tf.identity(biased9)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if use_camera:
            camera = cv2.VideoCapture(0)
        else:
            files = _iterate_files()

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600, 600)

        while not use_camera or camera.isOpened():
            start_time = time.time()

            choice = cv2.waitKey(1)
            if choice == 27:
                break

            if use_camera:
                ok, orig_img = camera.read()
                if not ok:
                    continue
            else:
                img_file = next(files)
                print(img_file)
                orig_img = cv2.imread(img_file)
                if orig_img is None:
                    continue

            img = tfnet.framework.resize_input(orig_img)
            img = np.expand_dims(img, 0)
            h, w, _ = orig_img.shape

            out = sess.run(output, {inpt: img})[0]

            threshold = 0.5
            boxes = tfnet.framework.findboxes(out)
            for box in boxes:
                box = tfnet.framework.process_box(box, h, w, threshold)
                if not box:
                    continue
                cv2.rectangle(
                    orig_img, (box[0], box[2]), (box[1], box[3]), (0, 255, 0), 3)
                cv2.putText(orig_img, box[4], (box[0], box[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('image', orig_img)

            print(time.time() - start_time)

            if not use_camera:
                time.sleep(1)
