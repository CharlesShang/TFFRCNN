import numpy as np
import tensorflow as tf

from ..roi_pooling_layer import roi_pooling_op as roi_pool_op
from ..rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from ..rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from ..rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py



DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print "assign pretrain model "+subkey+ " to "+key
                    except ValueError:
                        print "ignore "+key
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        return tf.reshape(tf.py_func(proposal_layer_py,\
                                     [input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales],\
                                     [tf.float32]),
                          [-1,5],name =name)


    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0],input[1],input[2],input[3],input[4], _feat_stride, anchor_scales],
                           [tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels') # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets') # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights') # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights') # shape is (1 x H x W x A, 4)


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            #inputs: 'rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas'
            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights \
                = tf.py_func(proposal_target_layer_py,
                             [input[0],input[1],input[2],input[3],classes],
                             [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
            # rois <- (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
            # rois = tf.convert_to_tensor(rois, name='rois')
            rois = tf.reshape(rois, [-1, 5], name='rois') # goes to roi_pooling
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels') # goes to FRCNN loss
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets') # goes to FRCNN loss
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

            self.layers['rois'] = rois

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, H, W, A x 2) -> (1, A x 2, H, W)
            # reshape: (1, 2, A x H, W)
            # transpose: -> (1, 9H, W, 2)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input,\
                               [input_shape[0],\
                                input_shape[1], \
                                -1,\
                                int(d)])

    # @layer
    # def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
    #     return feature_extrapolating_op.feature_extrapolating(input,
    #                           scales_base,
    #                           num_scale_base,
    #                           num_per_octave,
    #                           name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)


    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def smooth_l1_dist(self, predicts, targets, th = 0.1, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas = predicts - targets
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, th), tf.float32)
            return tf.square(deltas) * 0.5 * smoothL1_sign + \
                        (deltas_abs - th*th*0.5) * tf.abs(smoothL1_sign - 1)



    def build_loss(self):
        ############# RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        # ignore_label(-1)
        keeps_cls = tf.where(tf.not_equal(rpn_label, -1))
        keeps_rgs = tf.where(tf.equal(rpn_label, 1))
        rpn_cls_score = tf.gather(rpn_cls_score, keeps_cls)
        rpn_label = tf.gather(rpn_label, keeps_cls)
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score, rpn_label))

        # bounding box regression L1 loss
        # transpose to 1 * height * width * (anchor_nums * 4)
        rpn_bbox_pred = tf.reshape(self.get_output('rpn_bbox_pred'), [-1, 4])
        rpn_bbox_targets = tf.reshape(self.get_output('rpn-data')[1],[-1, 4])
        rpn_bbox_inside_weights = tf.reshape(self.get_output('rpn-data')[2], [-1, 4])
        rpn_bbox_outside_weights = tf.reshape(self.get_output('rpn-data')[3],[-1, 4]) # (1, H, W, A * 4)

        rpn_bbox_pred = tf.gather(rpn_bbox_pred, keeps_rgs)
        rpn_bbox_targets = tf.gather(rpn_bbox_targets, keeps_rgs)
        rpn_bbox_inside_weights = tf.gather(rpn_bbox_inside_weights, keeps_rgs)

        # smooth l1 loss, normalized by locations
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum( \
            rpn_bbox_inside_weights * self.smooth_l1_dist(rpn_bbox_pred, rpn_bbox_targets),\
            reduction_indices=[1])) * 10


        ############# R-CNN
        # classification loss
        # shape is batch * nclass
        cls_score = self.get_output('cls_score')
        label = tf.reshape(self.get_output('roi-data')[1], [-1]) # (HxWxA)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(cls_score, label))

        # bounding box regression L1 loss
        # shape is batch * (nclass * 4)
        bbox_pred = self.get_output('bbox_pred') # (HxW, Ax4)
        bbox_targets = self.get_output('roi-data')[2] # (HxW, Ax4)
        # each element is {0, 1}, represents background (0), objects (1)
        bbox_inside_weights = self.get_output('roi-data')[3] # (HxW, Ax5)
        bbox_outside_weights = self.get_output('roi-data')[4] # (HxW, Ax5)

        deltas = bbox_inside_weights * self.smooth_l1_dist(bbox_pred, bbox_targets)

        keeps_rois = tf.where(tf.not_equal(label, 0))
        deltas = tf.gather(tf.reshape(deltas, [-1, 4]), keeps_rois)

        # l1 distance
        loss_box = tf.reduce_mean(tf.reduce_sum(deltas, reduction_indices=[1])) * 20
        loss_box = tf.clip_by_value(loss_box, 0.0, 1)

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        return loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box