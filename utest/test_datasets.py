import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.datasets.factory import get_imdb
from lib.fast_rcnn.train import get_training_roidb
from lib.fast_rcnn.config import cfg, get_output_dir
from lib.roi_data_layer.layer import RoIDataLayer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)
    output_dir = get_output_dir(imdb, None)

    data_layer = RoIDataLayer(roidb, 3)
    blobs = data_layer.forward()

