# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .VGGnet_train import VGGnet_train
from .VGGnet_test import VGGnet_test
from .Resnet50_train import Resnet50_train
from .Resnet50_test import Resnet50_test
from .Resnet101_train import Resnet101_train
from .Resnet101_test import Resnet101_test
from . import factory
