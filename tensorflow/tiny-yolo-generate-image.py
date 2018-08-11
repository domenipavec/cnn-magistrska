#!/usr/bin/env python3
import io
import subprocess

import numpy as np
import cv2
import tensorflow as tf

from darkflow.net.build import TFNet

image = '../sample_img/sample_computer.jpg'
cnn_c_file = "../vivado/cnn/cnn.sdk/cnnGeneral/src/image.h"
cnn_software_c_file = "../vivado/cnn_software/cnn_software.sdk/cnn_software/src/image.h"

tfnet = TFNet({"model": "tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights"})

last_layer_size = 11

def resize_input(im):
    w = 32*last_layer_size
    h = w
    imsz = cv2.resize(im, (w, h))
    imsz = imsz[:, :, ::-1]
    return imsz

def number_array(values):
    return "{" + ", ".join(map(str, values)) + "}"

with tfnet.graph.as_default():
    layers = tfnet.darknet.layers

    def _get_var(layer, name):
        return tf.get_variable(
            'layer_%d_%s' % (layer, name),
            shape=layers[layer].wshape[name],
            initializer=layers[layer].w[name],
        )

    inpt = tf.placeholder(tf.float32, shape=(1, 32*last_layer_size, 32*last_layer_size, 3))

    conv1 = tf.nn.conv2d(
        inpt, layers[0].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch1 = tf.nn.batch_normalization(
        conv1,
        _get_var(0, 'moving_mean'),
        _get_var(0, 'moving_variance'),
        offset=layers[0].w['biases'],
        scale=_get_var(0, 'gamma'),
        variance_epsilon=1e-5,
    )

    leaky1 = tf.nn.leaky_relu(batch1, alpha=.1)

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
        offset=layers[3].w['biases'],
        scale=_get_var(3, 'gamma'),
        variance_epsilon=1e-5,
    )

    leaky2 = tf.nn.leaky_relu(batch2, alpha=.1)

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
        offset=layers[6].w['biases'],
        scale=_get_var(6, 'gamma'),
        variance_epsilon=1e-5,
    )

    leaky3 = tf.nn.leaky_relu(batch3, alpha=.1)

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
        offset=layers[9].w['biases'],
        scale=_get_var(9, 'gamma'),
        variance_epsilon=1e-5,
    )

    leaky4 = tf.nn.leaky_relu(batch4, alpha=.1)

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
        offset=layers[12].w['biases'],
        scale=_get_var(12, 'gamma'),
        variance_epsilon=1e-5,
    )

    leaky5 = tf.nn.leaky_relu(batch5, alpha=.1)

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
        offset=layers[15].w['biases'],
        scale=_get_var(15, 'gamma'),
        variance_epsilon=1e-5,
    )

    leaky6 = tf.nn.leaky_relu(batch6, alpha=.1)

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
        offset=layers[18].w['biases'],
        scale=_get_var(18, 'gamma'),
        variance_epsilon=1e-5,
    )

    leaky7 = tf.nn.leaky_relu(batch7, alpha=.1)

    conv8 = tf.nn.conv2d(
        leaky7, layers[20].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    batch8 = tf.nn.batch_normalization(
        conv8,
        _get_var(20, 'moving_mean'),
        _get_var(20, 'moving_variance'),
        offset=layers[20].w['biases'],
        scale=_get_var(20, 'gamma'),
        variance_epsilon=1e-5,
    )

    leaky8 = tf.nn.leaky_relu(batch8, alpha=.1)

    conv9 = tf.nn.conv2d(
        leaky8, layers[22].w['kernel'],
        strides=[1, 1, 1, 1], padding='SAME')

    biased9 = tf.nn.bias_add(conv9, layers[22].w['biases'])

    output = tf.identity(biased9)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        orig_img = cv2.imread(image)

        img8bit = resize_input(orig_img)
        img = img8bit/256
        img = np.expand_dims(img, 0)

        out = sess.run(output, {inpt: img})[0]
        if out.shape[-1] == 125:
            out = np.pad(out, ((0, 0), (0, 0), (0, 3)), 'constant')
            print(out.shape)

        print(np.max(sess.run(maxpool1, {inpt: img})))
        print(np.max(sess.run(maxpool2, {inpt: img})))
        print(np.max(sess.run(maxpool3, {inpt: img})))
        print(np.max(sess.run(maxpool4, {inpt: img})))
        print(np.max(sess.run(maxpool5, {inpt: img})))
        print(np.max(sess.run(maxpool6, {inpt: img})))

        with open("img.dat", "w") as output_file:
            np.savetxt(output_file, [32*last_layer_size, 3], fmt='%d')
            #  np.savetxt(output_file, img.flatten())
            np.savetxt(output_file, img8bit.flatten(), fmt='%d')
            np.savetxt(output_file, out.flatten())

        with open(cnn_c_file, "w") as cout:
            cout.write("unsigned char img[] = {};\n".format(number_array(img8bit.flatten())))
            cout.write("float expected[] = {};\n".format(number_array(out.flatten())))

        with open(cnn_software_c_file, "w") as cout:
            cout.write("unsigned char img[] = {};\n".format(number_array(img8bit.flatten())))
            cout.write("float expected[] = {};\n".format(number_array(out.flatten())))

