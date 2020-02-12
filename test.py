import argparse
import tensorflow as tf
import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils import anchor, box_utils, session_config
from utils.image_utils import ImageVisualizer
from models import network
from configs import test_config
config = test_config.Config()
if config.dataset_name == 'SED-dataset':
    from data_loader import sed_dataset as dataset

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

NUM_CLASSES = len(config.label_set)+1

def predict(imgs, default_boxes):
    confs, locs = ssd(imgs)
    #confs = tf.squeeze(confs, 0)
    #locs = tf.squeeze(locs, 0)

    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)
    boxes = box_utils.decode(default_boxes, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, NUM_CLASSES):
        cls_scores = confs[:, :, c]
        score_idx = cls_scores > 0.6
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = box_utils.compute_nms(cls_boxes, cls_scores, 0.45, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


if __name__ == '__main__':
    session_config.setup_gpus(True, 0.9)

    test_generator, test_length = dataset.Dataset().load_data_generator('train', num_examples = config.num_examples)

    try:
        ssd = network.create_ssd(NUM_CLASSES, config.arch,
                         config.pretrained_type,
                         config.checkpoint_dir,
                         config.checkpoint_path)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    os.makedirs('outputs/images', exist_ok=True)
    os.makedirs('outputs/detects', exist_ok=True)
    visualizer = ImageVisualizer(config.label_set, save_dir='outputs/images')

    for i, (filename, imgs, gt_confs, gt_locs) in enumerate(
        tqdm(test_generator, total=test_length,
             desc='Testing...', unit='images')):
        default_boxes = anchor.generate_default_boxes(config)
        boxes, classes, scores = predict(imgs, default_boxes)
        filename = filename.numpy()[0].decode()
        original_image = Image.open(
            os.path.join(config.image_dir, '{}'.format(filename)))
        boxes *= original_image.size * 2

        visualizer.save_image(
            original_image, boxes, classes, '{}'.format(filename))

        log_file = os.path.join('outputs/detects', '{}.txt')

        for cls, box, score in zip(classes, boxes, scores):
            cls_name = config.label_set[cls - 1]
            with open(log_file.format(cls_name), 'a') as f:
                f.write('{} {} {} {} {} {}\n'.format(
                    filename,
                    score,
                    *[coord for coord in box]))
