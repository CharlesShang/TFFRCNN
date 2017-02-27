# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from .VGGnet_test import VGGnet_test
from .VGGnet_testold import VGGnet_testold
from .VGGnet_train import VGGnet_train
from .Resnet50_test import Resnet50_test
from .Resnet50_train import Resnet50_train
from .Resnet101_test import Resnet101_test
from .Resnet101_train import Resnet101_train
from .PVAnet_train import PVAnet_train
from .PVAnet_test import PVAnet_test


def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
           return VGGnet_test()
        elif name.split('_')[1] == 'train':
           return VGGnet_train()
        elif name.split('_')[1] == 'testold':
            return VGGnet_testold()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    elif name.split('_')[0] == 'Resnet50':
        if name.split('_')[1] == 'test':
            return Resnet50_test()
        elif name.split('_')[1] == 'train':
            return Resnet50_train()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))
    elif name.split('_')[0] == 'Resnet101':
        if name.split('_')[1] == 'test':
            return Resnet101_test()
        elif name.split('_')[1] == 'train':
            return Resnet101_train()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))
    elif name.split('_')[0] == 'PVAnet':
        if name.split('_')[1] == 'test':
           return PVAnet_test()
        elif name.split('_')[1] == 'train':
           return PVAnet_train()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
