#!/usr/bin/env bash

python ./faster_rcnn/test_net.py \
--gpu 0 \
--weights ./output/faster_rcnn_end2end/voc_2007_trainval/VGGnet_fast_rcnn_iter_80000.ckpt \
--imdb voc_2007_test \
--cfg ./experiments/cfgs/faster_rcnn_end2end.yml \
--network VGGnet_test


python ./faster_rcnn/test_net.py \
--gpu 0 \
--weights ./output/faster_rcnn_voc_vgg/voc_0712_trainval/VGGnet_fast_rcnn_iter_100000.ckpt \
--imdb voc_2007_test \
--cfg ./experiments/cfgs/faster_rcnn_end2end.yml \
--network VGGnet_test

python ./faster_rcnn/test_net.py \
--gpu 0 \
--weights ./data/pretrain_model/Resnet_iter_200000.ckpt \
--imdb voc_2007_test \
--cfg ./experiments/cfgs/faster_rcnn_end2end_resnet.yml \
--network Resnet50_test



python ./faster_rcnn/test_net.py \
--gpu 0 \
--weights ./output/faster_rcnn_end2end_resnet_voc/voc_0712_trainval/Resnet50_iter_70000.ckpt \
--imdb voc_0712_test \
--cfg ./experiments/cfgs/faster_rcnn_end2end_resnet.yml \
--network Resnet50_test