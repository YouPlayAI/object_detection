class Config():
    def __init__(self):
        self.image_dir = '/home/dan/1T/SED-project/sed-datasets/SHWD/VOC2028/JPEGImages'
        self.dataset_dir = '/home/dan/1T/SED-project/sed-datasets'
        self.coco_api = '/home/dan/1T/cocoapi/PythonAPI'
        self.label_set = ['head', 'helmet']
        self.input_shape = [256, 256]
        self.batch_size = 64
        self.SSD : {ratios: [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                       scales: [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
                       fm_sizes: [38, 19, 10, 5, 3, 1],
                       image_size: 300}
        self.SSD512 : {ratios: [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
                       scales: [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                       fm_sizes: [64, 32, 16, 8, 6, 4, 1],
                       image_size: 512}