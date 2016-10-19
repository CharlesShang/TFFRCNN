# TFFRCNN

This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network (RPN).
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

### Acknowledgments: 

1. [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

2. [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)

3. [tensorflow-fast-rcnn](https://github.com/zplizzi/tensorflow-fast-rcnn)

### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `python-opencv`, `easydict` (recommend to install: [Anaconda](https://www.continuum.io/downloads))

### Requirements: hardware

1. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  git clone https://github.com/CharlesShang/TFFRCNN.git
  ```

2. Build the Cython modules
    ```Shell
    cd ./lib
    make # compile cython and roi_pooling_op, you may need to modify make.sh for your platform
    ```

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
python ./faster_rcnn/demo.py --model model_path
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Download list

1. [VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM)

2. [VGG16 - FasterRCN](https://drive.google.com/open?id=0ByuDEGFYmWsbZ0EzeUlHcGFIVWM).

### Training on Pascal VOC 2012

TODO