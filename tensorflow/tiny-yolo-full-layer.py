#!/usr/bin/env python3
import io
import subprocess

import numpy as np
import cv2
import tensorflow as tf

from darkflow.net.build import TFNet

image = '../sample_img/sample_computer.jpg'

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

    mean1 = _get_var(0, 'moving_mean')
    variance1 = _get_var(0, 'moving_variance')
    gamma1 = _get_var(0, 'gamma')
    bias1 = layers[0].w['biases']

    batch1 = tf.nn.batch_normalization(
        conv1,
        mean1,
        variance1,
        offset=bias1,
        scale=gamma1,
        variance_epsilon=1e-5,
    )

    leaky1 = tf.nn.leaky_relu(batch1, alpha=.1)

    maxpool1 = tf.nn.max_pool(
        leaky1,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
    )

    output = tf.identity(maxpool1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        orig_img = cv2.imread(image)

        img = tfnet.framework.resize_input(orig_img)
        img = np.expand_dims(img, 0)
        h, w, _ = orig_img.shape

        weights = sess.run(layers[0].w['kernel'])
        out = sess.run(output, {inpt: img})[0]

        gamma = sess.run(gamma1)
        variance = sess.run(variance1)
        bias = sess.run(bias1)
        mean = sess.run(mean1)

        max_diff = 0
        total_diff = 0
        total_count = 0
        for i in range(weights.shape[-1]):
            input_buffer = io.StringIO()
            scale = gamma[i]/np.sqrt(variance[i]+1e-5)
            add = bias[i] - mean[i]*scale
            np.savetxt(input_buffer, [scale, add])
            np.savetxt(input_buffer, weights[:, :, :, i].flatten())
            np.savetxt(input_buffer, img.flatten())

            print("Running layer", i)
            cmd = subprocess.run(
                "../hls/cnn_full_layer/solution1/csim/build/csim.exe",
                input=input_buffer.getvalue(),
                stdout=subprocess.PIPE,
                encoding='utf-8',
            )
            output = io.StringIO(cmd.stdout)

            for expected in np.nditer(out[:, :, i]):
                result = float(next(output))
                diff = abs(result-expected)
                max_diff = max(diff, max_diff)
                total_diff += diff
                total_count += 1

        print("Diff avg: %.2e max: %.2e" % (total_diff/total_count, max_diff))
