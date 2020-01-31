class Config():
    def __init__(self):
        self.dataset_name = 'SED-dataset'
        self.image_dir = '/home/docker_sharing_folder/sed-datasets/SHWD/VOC2028/JPEGImages'
        self.dataset_dir = '/home/docker_sharing_folder/sed-datasets'
        self.coco_api = '/home/docker_sharing_folder/cocoapi/PythonAPI'
        self.label_set = ['head', 'helmet']
        self.input_shape = [300, 300]
        self.num_examples = -1
        self.batch_size = 16
        self.SSD300 = {'ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                       'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
                       'fm_sizes': [38, 19, 10, 5, 3, 1],
                       'image_size': 300}
        self.SSD512 = {'ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
                       'scales': [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                       'fm_sizes': [64, 32, 16, 8, 6, 4, 1],
                       'image_size': 512}
        self.arch='ssd300'
        self.neg_ratio = 3
        self.initial_lr = 1e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.num_epochs = 120
        self.checkpoint_dir = 'checkpoints'
        self.pretrained_type = 'base'
        self.gpu_id = "0"
