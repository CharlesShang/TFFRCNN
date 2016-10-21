#!/bin/bash

python -m cProfile -o experiments/profiling/profile.out ./faster_rcnn/train_net.py\
 --gpu 0 --weights data/pretrain_model/VGG_imagenet.npy --imdb voc_2007_trainval \
 --iters 1000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_train

# generate an image
if [ ! -f experiments/profiling/gprof2dot.py ]; then 
	echo "Downloading ... "
	wget https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py -O experiments/profiling/gprof2dot.py
fi
python experiments/profiling/gprof2dot.py -f pstats experiments/profiling/profile.out | dot -Tpng -o experiments/profiling/profile.png