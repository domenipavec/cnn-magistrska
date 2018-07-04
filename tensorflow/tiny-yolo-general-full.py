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
        print(img.shape)

        weights = sess.run(layers[0].w['kernel'])
        out = sess.run(output, {inpt: img})[0]

        gamma = sess.run(gamma1)
        variance = sess.run(variance1)
        bias = sess.run(bias1)
        mean = sess.run(mean1)

        max_diff = 0
        total_diff = 0
        total_count = 0

        out_layers = weights.shape[-1]

        scale = [gamma[i]/np.sqrt(variance[i]+1e-5) for i in range(weights.shape[-1])]
        add = [bias[i] - mean[i]*scale[i] for i in range(weights.shape[-1])]

        input_buffer = io.StringIO()
        np.savetxt(input_buffer, [416, weights.shape[-2], out_layers, 0, 2], fmt='%d')
        np.savetxt(input_buffer, [(scale[i], add[i]) for i in range(out_layers)])
        for i in range(out_layers):
            np.savetxt(input_buffer, weights[:, :, :, i].flatten())
        np.savetxt(input_buffer, img.flatten())

        print("running")
        cmd = subprocess.run(
            "../hls/cnn_general/solution1/csim/build/csim.exe",
            input=input_buffer.getvalue(),
            stdout=subprocess.PIPE,
            encoding='utf-8',
        )
        output = io.StringIO(cmd.stdout)


        it = np.nditer(out[:, :, :out_layers], flags=['multi_index'])
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
