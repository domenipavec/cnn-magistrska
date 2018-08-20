#!/usr/bin/env python3
import os
from xml.etree import ElementTree

from darkflow.net.build import TFNet

annotations = '../VOC2007/Annotations/'
use_camera = False


def _iterate_files():
    for subdir, dirs, files in os.walk(annotations):
        for file in files:
            yield os.path.join(subdir, file)


tfnet = TFNet({"model": "tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights"})
labels = tfnet.meta['labels']

for fn in _iterate_files():
    print('Starting ' + fn)
    tree = ElementTree.parse(fn)
    print('Parsed')
    with open('groundtruths/{}.txt'.format(fn.split('/')[-1][:-4]), 'w') as fout:
        for obj in tree.iter('object'):
            cls = obj.find('name').text
            if cls not in labels:
                continue

            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text

            fout.write(' '.join([cls, xmin, ymin, xmax, ymax]) + '\n')
    print('Done')
