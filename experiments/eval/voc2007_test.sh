#!/usr/bin/env bash

python ./faster_rcnn/test_net.py \
--gpu 0 \
--weights ./output/faster_rcnn_end2end/voc_2007_trainval/VGGnet_fast_rcnn_iter_65000.ckpt \
--imdb voc_2007_test \
--cfg ./experiments/cfgs/faster_rcnn_end2end.yml \
--network VGGnet_test