#!/bin/bash

python ./faster_rcnn/train_net.py \
--gpu 0 \
--weights ./data/pretrain_model/VGG_imagenet.npy \
--imdb kittivoc_train \
--iters 160000 \
--cfg ./experiments/cfgs/faster_rcnn_kitti.yml \
--network VGGnet_train \
--restore 1