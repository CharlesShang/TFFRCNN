#!/usr/bin/env bash

python ./faster_rcnn/test_net.py \
--gpu 0 \
--weights ./data/VGGnet_fast_rcnn_iter_70000.ckpt \
--imdb voc_2007_test \
--cfg ./experiments/cfgs/faster_rcnn_end2end.yml \
--network VGGnet_testold