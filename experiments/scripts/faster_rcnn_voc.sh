#!/usr/bin/env bash
#python ./faster_rcnn/train_net.py \
#--gpu 0 \
#--weights ./data/pretrain_model/VGG_imagenet.npy \
#--imdb voc_2007_trainval \
#--iters 100000 \
#--cfg ./experiments/cfgs/faster_rcnn_end2end.yml \
#--network VGGnet_train \
#--restore 0

python ./faster_rcnn/train_net.py \
--gpu 0 \
--weights ./data/pretrain_model/VGG_imagenet.npy \
--imdb voc_0712_trainval \
--iters 100000 \
--cfg ./experiments/cfgs/faster_rcnn_end2end.yml \
--network VGGnet_train \
--restore 1