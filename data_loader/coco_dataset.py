#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
import cv2
import sys
import tensorflow as tf
from config import coco_config
config = coco_config.Config()
sys.path.append(config.coco_api)
from pycocotools.coco import COCO
from utils import anchor, box_utils
from functools import partial

def set_id_to_label(label_set):
    id_to_label = {}
    for idx, label in enumerate(label_set):
        id_to_label[idx+1] = label
    return id_to_label

class Dataset():
    def __init__(self, arch):
        self.dataset_name = 'SED-dataset'
        self.image_dir = config.image_dir
        self.dataset_dir = config.dataset_dir
        self.id_to_label = set_id_to_label(config.label_set)
        self.input_shape = (config.SSD300['image_size'], config.SSD300['image_size'])
        self.default_boxes = anchor.generate_default_boxes(config)

        self.train_annotation_path = os.path.join(self.dataset_dir, 'annotations', 'CNSI-SED-TRAIN.json')
        self.val_annotation_path = os.path.join(self.dataset_dir, 'annotations', 'CNSI-SED-VAL.json')
        self.eval_annotation_path = os.path.join(self.dataset_dir, 'annotations', 'CNSI-SED-TEST.json')

    def __len__(self):
        return len(self.ids)

    def _get_image(self, image_id, coco):
        filename = coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(image_path)[:,:,::-1]
        image = cv2.resize(image, self.input_shape)
        image = (image / 127.0) - 1.0
        return filename, image

    def _get_annotation(self, image_id, cat_ids, coco):
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids, iscrowd=None)
        boxes = []
        for ann_id in ann_ids:
            x, y, w, h = coco.loadAnns(ann_id)[0]['bbox']
            xmin = x / w
            ymin = y / h
            xmax = (x+w)/w
            ymax = (y+h)/h
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
        labels = [ coco.loadAnns(ann_id)[0]['category_id'] for ann_id in ann_ids]
        return boxes, labels
    def generate(self, split, num_examples=-1):
        """
            num_examples : The number of examples to be used.
            It's used if you want to make model overfit a few examples
        """
        if split == 'train':
            coco = COCO(self.train_annotation_path)
        elif split == 'val':
            coco = COCO(self.val_annotation_path)
        elif split == 'test':
            coco = COCO(self.eval_annotation_path)
        else:
            raise ValueError("Wrong split name!")

        self.ids = coco.getImgIds()[:num_examples]
        cat_ids = coco.getCatIds(self.id_to_label.values())

        for image_id in self.ids:
            filename, image = self._get_image(image_id, coco)
            boxes, labels = self._get_annotation(image_id, cat_ids, coco)          
            image = tf.constant(image, dtype=tf.float32)
            gt_confs, gt_locs = box_utils.compute_target(self.default_boxes, boxes, labels)
            yield filename, image, gt_confs, gt_locs

    def load_data_generator(self, split='train', num_examples=-1):
        """
            num_examples : The number of examples to be used.
            It's used if you want to make model overfit a few examples
        """
        if split == 'train':
            gen = partial(self.generate, split, num_examples)
            dataset = tf.data.Dataset.from_generator(gen,
                (tf.string, tf.float32, tf.int32, tf.float32)).repeat().shuffle(1000)
        elif split == 'val':
            gen = partial(self.generate, split, num_examples)
            dataset = tf.data.Dataset.from_generator(gen,
                (tf.string, tf.float32, tf.int32, tf.float32))
        elif split == 'test':
            gen = partial(self.generate, split, num_examples)
            dataset = tf.data.Dataset.from_generator(gen,
                (tf.string, tf.float32, tf.int32, tf.float32))
        else:
            raise ValueError("Wrong split name!")

        return dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)

dataset = Dataset()
x = dataset.load_data_generator('train', 10)
for i, j, k, l in x.take(10):
    print(i.shape, j.shape, k.shape, l.shape)