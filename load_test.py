from configs import train_config
config = train_config.Config()
import sys
import os
sys.path.append(config.coco_api)
from pycocotools.coco import COCO
import cv2
from random import shuffle

train_annotation_path = os.path.join(config.dataset_dir, 'annotations', 'CNSI-SED-TRAIN.json')
coco = COCO(train_annotation_path)
ids = coco.getImgIds()
#ids = shuffle(coco.getImgIds())


print(ids)
