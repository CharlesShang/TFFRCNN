# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""
import numpy as np
import os
import tensorflow as tf
import cv2

from .nms_wrapper import nms_wrapper
from ..roi_data_layer.layer import RoIDataLayer
from ..utils.timer import Timer
from ..gt_data_layer import roidb as gdl_roidb
from ..roi_data_layer import roidb as rdl_roidb

# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
# <<<< obsolete

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, output_dir, logdir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.train.SummaryWriter(logdir=logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=10)

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred') and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))

    def build_image_summary(self):
        """
        A simple graph for write image summary
        :return:
        """
        log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
        log_image_name = tf.placeholder(tf.string)
        log_image = tf.image_summary(log_image_name, tf.expand_dims(log_image_data, 0), max_images=50)
        # log_image = tf.image_summary(log_image_name, log_image_data, max_images=50)
        return log_image, log_image_data, log_image_name


    def train_model(self, sess, max_iters, restore=False):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
   
        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        # ignore_label(-1)
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score, rpn_label))


        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])
        smoothL1_sign = tf.cast(tf.less(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),1),tf.float32)
        rpn_loss_box = tf.mul(tf.reduce_mean(tf.reduce_sum(tf.mul(rpn_bbox_outside_weights,tf.add(
                       tf.mul(tf.mul(tf.pow(tf.mul(rpn_bbox_inside_weights, tf.sub(rpn_bbox_pred, rpn_bbox_targets))*3,2),0.5),smoothL1_sign),
                       tf.mul(tf.sub(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),0.5/9.0),tf.abs(smoothL1_sign-1)))), reduction_indices=[1,2])),10)

        # R-CNN
        # classification loss
        cls_score = self.net.get_output('cls_score')
        #label = tf.placeholder(tf.int32, shape=[None])
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(cls_score, label))


        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]
        loss_box = tf.reduce_mean(tf.reduce_sum(tf.mul(bbox_outside_weights,tf.mul(bbox_inside_weights, tf.abs(tf.sub(bbox_pred, bbox_targets)))), reduction_indices=[1]))


        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # scalar summary
        tf.scalar_summary('rpn_rgs_loss', rpn_loss_box)
        tf.scalar_summary('rpn_cls_loss', rpn_cross_entropy)
        tf.scalar_summary('cls_loss', cross_entropy)
        tf.scalar_summary('rgs_loss', loss_box)
        tf.scalar_summary('loss', loss)
        summary_op = tf.merge_all_summaries()

        # image writer
        # NOTE: this image is independent to summary_op
        log_image, log_image_data, log_image_name =\
            self.build_image_summary()

        # optimizer
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
        else:
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        tvars = tf.trainable_variables()
        grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1.0)
        train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)

        # intialize variables
        sess.run(tf.initialize_all_variables())

        # load vgg16
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)
        if restore:
            print 'Restoring from {}...'.format(self.output_dir),
            ckpt = tf.train.get_checkpoint_state(self.output_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print 'done'
            else: print 'failed.'

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # learning rate
            if iter >= cfg.TRAIN.STEPSIZE:
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
            else:
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))

            # get one batch
            timer.tic()

            blobs = data_layer.forward()

            if (iter + 1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'image: %s' %(blobs['im_name']),

            feed_dict={
                self.net.data: blobs['data'],
                self.net.im_info: blobs['im_info'],
                self.net.keep_prob: 0.5,
                self.net.gt_boxes: blobs['gt_boxes']}

            res_fetches = [self.net.get_output('cls_prob'),  # FRCNN class prob
                           self.net.get_output('bbox_pred'), # FRCNN rgs output
                           self.net.get_output('rois')]  # RPN rgs output

            fetch_list = [rpn_cross_entropy,
                          rpn_loss_box,
                          cross_entropy,
                          loss_box,
                          summary_op,
                          train_op] + res_fetches

            rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value,\
                summary_str, _, \
                cls_prob, bbox_pred, rois = \
                sess.run(fetches=fetch_list, feed_dict=feed_dict)

            self.writer.add_summary(summary=summary_str, global_step=global_step.eval())

            _diff_time = timer.toc(average=False)

            if (iter) % cfg.TRAIN.LOG_IMAGE_ITERS == 0:
                # plus mean
                ori_im = np.squeeze(blobs['data']) + cfg.PIXEL_MEANS
                ori_im = ori_im.astype(dtype=np.uint8, copy=False)
                # draw rects
                boxes, scores = _process_boxes_scores(cls_prob, bbox_pred, rois, blobs['im_info'][0][2], ori_im.shape)
                res = nms_wrapper(scores, boxes)
                image = _draw_boxes_to_image(ori_im, res)
                log_image_name_str = ('%06d_' % iter ) + blobs['im_name']
                log_image_summary_op = \
                    sess.run(log_image, \
                             feed_dict={log_image_name: log_image_name_str,\
                                        log_image_data: image})
                self.writer.add_summary(log_image_summary_op, global_step=global_step.eval())

            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,\
                         rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval())
                print 'speed: {:.3f}s / iter'.format(_diff_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            # TODO: fix multiscale training (single scale is already a good trade-off)
            print ('#### warning: multi-scale has not been tested.')
            print ('#### warning: using single scale by setting IS_MULTISCALE: False.')
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            # obsolete
            # layer = GtDataLayer(roidb)
            raise "Calling caffe modules..."
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def _process_boxes_scores(cls_prob, bbox_pred, rpn_rois, im_scale, im_shape):
    """
    process the output tensors, to get the boxes and scores
    """
    assert rpn_rois.shape[0] == bbox_pred.shape[0],\
        'Out tensor shapes dont match'
    boxes = rpn_rois[:, 1:5] / im_scale
    scores = cls_prob
    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape)
    else:
        # Simply repeat the boxes, once for each class
        boxes = np.tile(boxes, (1, scores.shape[1]))
    return pred_boxes, scores

def _draw_boxes_to_image(im, res):
    colors = [(86, 0, 240), (86, 67, 140), (0, 76, 255), \
              (151, 0, 255), (86, 255, 234), (0, 117, 255),\
              (58, 184, 14), (173, 225, 61), (121, 82, 6),\
              (174, 29, 128), (115, 154, 81), (243, 223, 48)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = np.copy(im)
    cnt = 0
    for r in res:
        if r['dets'] is None: continue
        dets = r['dets']
        for i in range(0, dets.shape[0]):
            (x1, y1, x2, y2, score) = dets[i, :]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[cnt], 2)
            text = '{:s} {:.2f}'.format(r['class'], score)
            cv2.putText(image, text, (x1, y1), font, 0.6, colors[cnt], 1)
            cnt = (cnt + 1) % len(colors)
    return image


def train_net(network, imdb, roidb, output_dir, log_dir, pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN network."""

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, logdir= log_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters, restore=restore)
        print 'done solving'
