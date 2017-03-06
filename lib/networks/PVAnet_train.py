# --------------------------------------------------------
# TFFRCNN - Resnet50
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by miraclebiu
# --------------------------------------------------------

import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg
import numpy as np


class PVAnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes, \
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]
        (self.feed('data')
         .pva_negation_block(7, 7, 16, 2, 2, name='conv1_1', negation=True)         # downsample
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')                       # downsample
         .conv(1, 1, 24, 1, 1, name='conv2_1/1/conv', biased=True, relu=False)
         .pva_negation_block_v2(3, 3, 24, 1, 1, 24, name='conv2_1/2', negation=False)
         .pva_negation_block_v2(1, 1, 64, 1, 1, 24, name='conv2_1/3', negation=True))

        (self.feed('pool1')
         .conv(1,1, 64, 1, 1, name='conv2_1/proj', relu=True))

        (self.feed('conv2_1/3', 'conv2_1/proj')
         .add(name='conv2_1')
         .pva_negation_block_v2(1, 1, 24, 1, 1, 64, name='conv2_2/1', negation = False)
         .pva_negation_block_v2(3, 3, 24, 1, 1, 24, name='conv2_2/2', negation = False)
         .pva_negation_block_v2(1, 1, 64, 1, 1, 24, name='conv2_2/3', negation = True))

        (self.feed('conv2_2/3', 'conv2_1')
         .add(name='conv2_2')
         .pva_negation_block_v2(1, 1, 24, 1, 1, 64, name='conv2_3/1', negation=False)
         .pva_negation_block_v2(3, 3, 24, 1, 1, 24, name='conv2_3/2', negation=False)
         .pva_negation_block_v2(1, 1, 64, 1, 1, 24, name='conv2_3/3', negation=True))

        (self.feed('conv2_3/3', 'conv2_2')
         .add(name='conv2_3')
         .pva_negation_block_v2(1, 1, 48, 2, 2, 64, name='conv3_1/1', negation=False) # downsample
         .pva_negation_block_v2(3, 3, 48, 1, 1, 48, name='conv3_1/2', negation=False)
         .pva_negation_block_v2(1, 1, 128, 1, 1, 48, name='conv3_1/3', negation=True))

        (self.feed('conv3_1/1/relu')
         .conv(1, 1, 128, 2, 2, name='conv3_1/proj', relu=True))

        (self.feed('conv3_1/3', 'conv3_1/proj')  # 128
         .add(name='conv3_1')
         .pva_negation_block_v2(1, 1, 48, 1, 1, 128, name='conv3_2/1', negation=False)
         .pva_negation_block_v2(3, 3, 48, 1, 1, 48, name='conv3_2/2', negation=False)
         .pva_negation_block_v2(1, 1, 128, 1, 1, 48, name='conv3_2/3', negation=True))

        (self.feed('conv3_2/3', 'conv3_1')  # 128
         .add(name='conv3_2')
         .pva_negation_block_v2(1, 1, 48, 1, 1, 128, name='conv3_3/1', negation=False)
         .pva_negation_block_v2(3, 3, 48, 1, 1, 48, name='conv3_3/2', negation=False)
         .pva_negation_block_v2(1, 1, 128, 1, 1, 48, name='conv3_3/3', negation=True))

        (self.feed('conv3_3/3', 'conv3_2')  # 128
         .add(name='conv3_3')
         .pva_negation_block_v2(1, 1, 48, 1, 1, 128, name='conv3_4/1', negation=False)
         .pva_negation_block_v2(3, 3, 48, 1, 1, 48, name='conv3_4/2', negation=False)
         .pva_negation_block_v2(1, 1, 128, 1, 1, 48, name='conv3_4/3', negation=True))

        (self.feed('conv3_4/3', 'conv3_3')  # 128
         .add(name='conv3_4')
         .max_pool(3, 3, 2, 2, padding='SAME', name='downsample')) # downsample

        (self.feed('conv3_4')
         .pva_inception_res_block(name = 'conv4_4', name_prefix = 'conv4_', type='a') # downsample
         .pva_inception_res_block(name='conv5_4', name_prefix='conv5_', type='b')     # downsample
         .batch_normalization(name='conv5_4/last_bn', relu=False)
         .relu(name='conv5_4/last_relu'))

        (self.feed('conv5_4/last_relu')
         .upconv(tf.shape(self.layers['downsample']),
                 384, 4, 2, name = 'upsample', biased= False, relu=False, trainable=True)) # upsample

        (self.feed('downsample', 'conv4_4')
         .concat(axis=3, name='concat'))

        # ========= RPN ============
        (self.feed('concat')
         .conv(1, 1, 128, 1, 1, name='convf_rpn', biased=True, relu=True)
         .conv(3, 3, 384, 1, 1, name='rpn_conv/3x3', biased=True, relu=True)
         .conv(1, 1, len(anchor_scales) * 3 * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
         .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))
        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred'))

        # ========= RoI Proposal ============
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(anchor_scales) * 3 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name='rpn_rois'))

        (self.feed('rpn_rois', 'gt_boxes', 'gt_ishard', 'dontcare_areas')
         .proposal_target_layer(n_classes, name='roi-data'))

        # ========= RCNN ============
        (self.feed('concat')
         .conv(1, 1, 384, 1, 1, name='convf_2', biased=True, relu=True))

        # (self.feed('convf_rpn', 'convf_2')
        #  .concat(axis=3, name='convf'))

        (self.feed('convf_2', 'roi-data')
         .roi_pool(6, 6, 1.0 / 16, name='roi_pooling')
         .fc(4096, name='fc6', relu=False)
         .bn_scale_combo(c_in = 4096, name='fc6', relu=True)
         .dropout(0.5, name='fc6/drop6')
         .fc(4096, name='fc7', relu=False)
         .bn_scale_combo(c_in=4096, name='fc7', relu=True)
         .dropout(0.5, name='fc7/drop7')
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob'))

        (self.feed('fc7/drop7')
         .fc(n_classes * 4, relu=False, name='bbox_pred'))

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        # print (data_dict.keys())
        for key in sorted(data_dict.keys(), reverse=True):
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print "assign pretrain model " + subkey + " to " + key + '/' + subkey
                    except ValueError:
                        print "ignore " + key + '/' + subkey + ' shape:', data_dict[key][subkey].shape
                        if not ignore_missing:

                            raise