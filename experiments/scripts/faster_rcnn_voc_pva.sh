#!/usr/bin/env bash

python ./faster_rcnn/train_net.py \
--gpu 0 \
--weights ./data/pretrain_model/pva9.1_preAct_train_iter_1900000.npy \
--imdb voc_0712_trainval \
--iters 160000 \
--cfg ./experiments/cfgs/faster_rcnn_end2end_pva.yml \
--network PVAnet_train \
--restore 0
