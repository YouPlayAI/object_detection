#!/usr/bin/python3
# coding=utf-8
import os
import numpy as np
import cv2
import sys
from random import shuffle
import tensorflow as tf
from functools import partial

from utils import anchor
from utils import box_utils
from configs import train_config
config = train_config.Config()
sys.path.append(config.coco_api)
from pycocotools.coco import COCO

def set_id_to_label(label_set):
    id_to_label = {}
    for idx, label in enumerate(label_set):
        id_to_label[idx+1] = label
    return id_to_label

class Dataset():
    def __init__(self):
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
        image_info = coco.loadImgs(image_id)[0]
        filename = image_info['file_name']
        original_size = (int(image_info['height']), int(image_info['width']))
        image_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            filename = os.path.splitext(filename)[0] + '.JPG'
            image_path = os.path.join(self.image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Can't not find correct filepath", filename)
            print(filename)
        image = image[:,:,::-1]
        image = cv2.resize(image, self.input_shape)
        image = (image / 127.0) - 1.0
        return filename, image, original_size

    def _get_annotation(self, image_id, cat_ids, original_size, coco):
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids, iscrowd=None)
        boxes = []
        (image_height, image_width) = original_size[0], original_size[1]
        for ann_id in ann_ids:
            x, y, w, h = coco.loadAnns(ann_id)[0]['bbox']
            xmin = x / image_width
            ymin = y / image_height
            xmax = (x+w)/image_width
            ymax = (y+h)/image_height
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
        labels = [ coco.loadAnns(ann_id)[0]['category_id'] for ann_id in ann_ids]
        return boxes, labels

    def generate(self, split, coco, ids, cat_ids, num_examples=-1):
        """
            num_examples : The number of examples to be used.
            It's used if you want to make model overfit a few examples
        """
        for image_id in ids:
            filename, image, original_size = self._get_image(image_id, coco)
            boxes, labels = self._get_annotation(image_id, cat_ids, original_size, coco)
            image = tf.constant(image, dtype=tf.float32)
            gt_confs, gt_locs = box_utils.compute_target(self.default_boxes, boxes, labels)
            yield filename, image, gt_confs, gt_locs
    def load_data_generator(self, split, batch_size, num_examples=-1):
        """
            num_examples : The number of examples to be used.
            It's used if you want to make model overfit a few examples
        """
        if split == 'train':
            coco = COCO(self.train_annotation_path)
            ids = coco.getImgIds()[:num_examples]
            shuffle(ids)
            cat_ids = coco.getCatIds(self.id_to_label.values())
            gen = partial(self.generate, split, coco, ids, cat_ids, num_examples)

            if len(ids) > 1000:
                shuffle_buffer = 1000
            else:
                shuffle_buffer = len(ids)
            dataset = tf.data.Dataset.from_generator(gen,
                (tf.string, tf.float32, tf.int32, tf.float32)).shuffle(shuffle_buffer)
        elif split == 'val':
            coco = COCO(self.val_annotation_path)
            ids = coco.getImgIds()[:num_examples]
            shuffle(ids)
            cat_ids = coco.getCatIds(self.id_to_label.values())
            gen = partial(self.generate, split, coco, ids, cat_ids, num_examples)
            dataset = tf.data.Dataset.from_generator(gen,
                (tf.string, tf.float32, tf.int32, tf.float32))
        elif split == 'test':
            coco = COCO(self.eval_annotation_path)
            ids = coco.getImgIds()[:num_examples]
            shuffle(ids)
            cat_ids = coco.getCatIds(self.id_to_label.values())
            gen = partial(self.generate, split, coco, ids, cat_ids, num_examples)
            dataset = tf.data.Dataset.from_generator(gen,
                (tf.string, tf.float32, tf.int32, tf.float32))
        else:
            raise ValueError("Wrong split name!")

        return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), len(ids)
