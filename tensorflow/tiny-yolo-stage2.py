#!/usr/bin/env python3
import io
import subprocess

import numpy as np
import cv2
import tensorflow as tf

from darkflow.net.build import TFNet

image = '../sample_img/sample_computer.jpg'

tfnet = TFNet({"model": "tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights"})

last_layer_size = 11

def resize_input(im):
    w = 32*last_layer_size
    h = w
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return imsz



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

    mean7 = _get_var(18, 'moving_mean')
    variance7 = _get_var(18, 'moving_variance')
    bias7 = layers[18].w['biases']
    gamma7 = _get_var(18, 'gamma')

    batch7 = tf.nn.batch_normalization(
        conv7,
        mean7,
        variance7,
        offset=bias7,
        scale=gamma7,
        variance_epsilon=1e-5,
    )

    leaky7 = tf.nn.leaky_relu(batch7, alpha=.1)

    stage2input = tf.identity(maxpool6)
    output = tf.identity(leaky7)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        orig_img = cv2.imread(image)

        img = resize_input(orig_img)
        img = np.expand_dims(img, 0)
        h, w, _ = orig_img.shape

        weights = sess.run(layers[18].w['kernel'])
        stage2in = sess.run(stage2input, {inpt: img})
        out = sess.run(output, {inpt: img})[0]

        max_diff = 0
        total_diff = 0
        total_count = 0

        gamma = sess.run(gamma7)
        variance = sess.run(variance7)
        bias = sess.run(bias7)
        mean = sess.run(mean7)

        scale = [gamma[i]/np.sqrt(variance[i]+1e-5) for i in range(weights.shape[-1])]
        add = [bias[i] - mean[i]*scale[i] for i in range(weights.shape[-1])]

        input_buffer = io.StringIO()
        np.savetxt(input_buffer, [last_layer_size, weights.shape[-2], weights.shape[-1], 1, 0], fmt='%d')
        np.savetxt(input_buffer, [(scale[i], add[i]) for i in range(weights.shape[-1])])
        np.savetxt(input_buffer, stage2in.flatten())
        for i in range(weights.shape[-1]):
            np.savetxt(input_buffer, weights[:, :, :, i].flatten())

        print("running")
        cmd = subprocess.run(
            "../hls/cnn_general/solution1/csim/build/csim.exe",
            input=input_buffer.getvalue(),
            stdout=subprocess.PIPE,
            encoding='utf-8',
        )
        output = io.StringIO(cmd.stdout)

        for i in range(weights.shape[-1]):
            it = np.nditer(out[:, :, i], flags=['multi_index'])
            while not it.finished:
                expected = it[0]
                result = float(next(output))
                diff = abs(result-expected)
                if diff > 1e-3:
                    print(it.multi_index, result, diff)
                max_diff = max(diff, max_diff)
                total_diff += diff
                total_count += 1
                it.iternext()
            #  break

        print("Diff avg: %.2e max: %.2e" % (total_diff/total_count, max_diff))

        while 1:
            try:
                print(next(output))
            except StopIteration:
                break
