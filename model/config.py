import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    print('cur', cur_dir)
    this_dir_name = cur_dir.split('/')[-1]
    print('this', this_dir_name)
    root_dir = os.path.join(cur_dir, '..')
    print(root_dir)

    model = 'MLPE50'

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = list(range(6,50,6))

    batch_size = 2
    weight_decay = 1e-5

    num_class = 17
    img_path = os.path.join(root_dir, 'data', 'COCO2017', 'train2017')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    # data augmentation setting
    bbox_extend_factor = (0.1, 0.15) # x, y
    scale_factor=(0.7, 1.35)
    rot_factor=45

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (256, 192) #height, width
    output_shape = (64, 48) #height, width
    gaussain_kernel = (13, 13)
    #
    gk15 = (23, 23)
    gk11 = (17, 17)
    gk9 = (13, 13)
    gk7 = (9, 9)
    gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_train.json')
    print(gt_path)

cfg = Config()
add_pypath(cfg.root_dir)

