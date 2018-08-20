#!/usr/bin/env python3
import os
import io
import subprocess
import time

import numpy as np
import cv2
from darkflow.net.build import TFNet

images = '../VOC2007/JPEGImages/'

last_layer_size = 11


def _iterate_files():
    for subdir, dirs, files in os.walk(images):
        for file in files:
            yield os.path.join(subdir, file)


def resize_input(im):
    w = 32*last_layer_size
    h = w
    imsz = cv2.resize(im, (w, h))
    imsz = imsz/255.
    imsz = imsz[:, :, ::-1]
    return imsz


tfnet = TFNet({"model": "tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights"})

for i, img_file in enumerate(_iterate_files()):
    start_time = time.time()

    orig_img = cv2.imread(img_file)
    if orig_img is None:
        continue

    img = resize_input(orig_img)
    img = np.expand_dims(img, 0)
    h, w, _ = orig_img.shape

    input_buffer = io.StringIO()
    np.savetxt(input_buffer, [32*last_layer_size, 3], fmt='%d')
    np.savetxt(input_buffer, img.flatten())

    cmd = subprocess.run(
        "../hls/cnn_sim/solution1/csim/build/csim.exe",
        input=input_buffer.getvalue(),
        stdout=subprocess.PIPE,
        encoding='utf-8',
    )
    output = io.StringIO(cmd.stdout)

    out = np.reshape(np.fromiter(output, np.float32), (11, 11, 128))
    out = out[:, :, :125]
    out = out.copy(order='C')

    threshold = 0
    boxes = tfnet.framework.findboxes(out)

    with open('detections-fpga/{}.txt'.format(img_file.split('/')[-1][:-4]), 'w') as fout:
        for box in boxes:
            box = tfnet.framework.process_box(box, h, w, threshold)
            if not box:
                continue

            left, right, top, bottom, cls, idx, prob = box

            fout.write(' '.join(str(x) for x in [cls, prob, left, top, right, bottom]) + '\n')

    print(i, time.time() - start_time)