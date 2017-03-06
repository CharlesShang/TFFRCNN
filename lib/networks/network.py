import numpy as np
import tensorflow as tf

from ..fast_rcnn.config import cfg
from ..roi_pooling_layer import roi_pooling_op as roi_pool_op
from ..rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from ..rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from ..rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
# FCN pooling
from ..psroi_pooling_layer import psroi_pooling_op as psroi_pooling_op


DEFAULT_PADDING = 'SAME'

def include_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

@include_original
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

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    """
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
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
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
            """

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def upconv(self, input, shape, c_o, ksize=4, stride = 2, name = 'upconv', biased=False, relu=True, padding=DEFAULT_PADDING,
             trainable=True):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = tf.shape(input)
        if shape is None:
            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            h = ((in_shape[1] ) * stride)
            w = ((in_shape[2] ) * stride)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name) as scope:
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            filters = self.make_var('weights', filter_shape, init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape,
                                            strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv

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
    def psroi_pool(self, input, output_dim, group_size, spatial_scale, name):
        """contribution by miraclebiu"""
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        return psroi_pooling_op.psroi_pool(input[0], input[1],
                                           output_dim=output_dim,
                                           group_size=group_size,
                                           spatial_scale=spatial_scale,
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
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
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
        return tf.concat(axis=axis, values=inputs, name=name)

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

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
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
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1], name=name)

    @layer
    def batch_normalization(self,input,name,relu=True, is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

    @layer
    def negation(self, input, name):
        """ simply multiplies -1 to the tensor"""
        return tf.multiply(input, -1.0, name=name)

    @layer
    def bn_scale_combo(self, input, c_in, name, relu=True):
        """ PVA net BN -> Scale -> Relu"""
        with tf.variable_scope(name) as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            # alpha = tf.get_variable('bn_scale/alpha', shape=[c_in, ], dtype=tf.float32,
            #                     initializer=tf.constant_initializer(1.0), trainable=True,
            #                     regularizer=self.l2_regularizer(0.00001))
            # beta = tf.get_variable('bn_scale/beta', shape=[c_in, ], dtype=tf.float32,
            #                    initializer=tf.constant_initializer(0.0), trainable=True,
            #                    regularizer=self.l2_regularizer(0.00001))
            # bn = tf.add(tf.mul(bn, alpha), beta)
            if relu:
                bn = tf.nn.relu(bn, name='relu')
            return bn

    @layer
    def pva_negation_block(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True, padding=DEFAULT_PADDING, trainable=True,
                           scale = True, negation = True):
        """ for PVA net, Conv -> BN -> Neg -> Concat -> Scale -> Relu"""
        with tf.variable_scope(name) as scope:
            conv = self.conv._original(self, input, k_h, k_w, c_o, s_h, s_w, biased=biased, relu=False, name='conv', padding=padding, trainable=trainable)
            conv = self.batch_normalization._original(self, conv, name='bn', relu=False, is_training=False)
            c_in = c_o
            if negation:
                conv_neg = self.negation._original(self, conv, name='neg')
                conv = tf.concat(axis=3, values=[conv, conv_neg], name='concat')
                c_in += c_in
            if scale:
                # y = \alpha * x + \beta
                alpha = tf.get_variable('scale/alpha', shape=[c_in,], dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0), trainable=True, regularizer=self.l2_regularizer(0.00001))
                beta = tf.get_variable('scale/beta', shape=[c_in, ], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.0), trainable=True, regularizer=self.l2_regularizer(0.00001))
                # conv = conv * alpha + beta
                conv = tf.add(tf.multiply(conv, alpha), beta)
            return tf.nn.relu(conv, name='relu')

    @layer
    def pva_negation_block_v2(self, input, k_h, k_w, c_o, s_h, s_w, c_in, name, biased=True, padding=DEFAULT_PADDING, trainable=True,
                           scale = True, negation = True):
        """ for PVA net, BN -> [Neg -> Concat ->] Scale -> Relu -> Conv"""
        with tf.variable_scope(name) as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            if negation:
                bn_neg = self.negation._original(self, bn, name='neg')
                bn = tf.concat(axis=3, values=[bn, bn_neg], name='concat')
                c_in += c_in
                # y = \alpha * x + \beta
                alpha = tf.get_variable('scale/alpha', shape=[c_in,], dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0), trainable=True, regularizer=self.l2_regularizer(0.00004))
                beta = tf.get_variable('scale/beta', shape=[c_in, ], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.0), trainable=True, regularizer=self.l2_regularizer(0.00004))
                bn = tf.add(tf.multiply(bn, alpha), beta)
            bn = tf.nn.relu(bn, name='relu')
            if name == 'conv3_1/1': self.layers['conv3_1/1/relu'] = bn

            conv = self.conv._original(self, bn, k_h, k_w, c_o, s_h, s_w, biased=biased, relu=False, name='conv', padding=padding,
                         trainable=trainable)
            return conv

    @layer
    def pva_inception_res_stack(self, input, c_in, name, block_start = False, type = 'a'):

        if type == 'a':
            (c_0, c_1, c_2, c_pool, c_out) = (64, 64, 24, 128, 256)
        elif type == 'b':
            (c_0, c_1, c_2, c_pool, c_out) = (64, 96, 32, 128, 384)
        else:
            raise ('Unexpected inception-res type')
        if block_start:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope(name+'/incep') as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            bn_scale = self.scale._original(self, bn, c_in, name='bn_scale')
            ## 1 x 1

            conv = self.conv._original(self, bn_scale, 1, 1, c_0, stride, stride, name='0/conv', biased = False, relu=False)
            conv_0 = self.bn_scale_combo._original(self, conv, c_in=c_0, name ='0', relu=True)

            ## 3 x 3
            bn_relu = tf.nn.relu(bn_scale, name='relu')
            if name == 'conv4_1': tmp_c = c_1; c_1 = 48
            conv = self.conv._original(self, bn_relu, 1, 1, c_1, stride, stride, name='1_reduce/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_1, name='1_reduce', relu=True)
            if name == 'conv4_1': c_1 = tmp_c
            conv = self.conv._original(self, conv, 3, 3, c_1 * 2, 1, 1, name='1_0/conv', biased = False, relu=False)
            conv_1 = self.bn_scale_combo._original(self, conv, c_in=c_1 * 2, name='1_0', relu=True)

            ## 5 x 5
            conv = self.conv._original(self, bn_scale, 1, 1, c_2, stride, stride, name='2_reduce/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_2, name='2_reduce', relu=True)
            conv = self.conv._original(self, conv, 3, 3, c_2 * 2, 1, 1, name='2_0/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_2 * 2, name='2_0', relu=True)
            conv = self.conv._original(self, conv, 3, 3, c_2 * 2, 1, 1, name='2_1/conv', biased = False, relu=False)
            conv_2 = self.bn_scale_combo._original(self, conv, c_in=c_2 * 2, name='2_1', relu=True)

            ## pool
            if block_start:
                pool = self.max_pool._original(self, bn_scale, 3, 3, 2, 2, padding=DEFAULT_PADDING, name='pool')
                pool = self.conv._original(self, pool, 1, 1, c_pool, 1, 1, name='poolproj/conv', biased = False, relu=False)
                pool = self.bn_scale_combo._original(self, pool, c_in=c_pool, name='poolproj', relu=True)

        with tf.variable_scope(name) as scope:
            if block_start:
                concat = tf.concat(axis=3, values=[conv_0, conv_1, conv_2, pool], name='concat')
                proj = self.conv._original(self, input, 1, 1, c_out, 2, 2, name='proj', biased=True,
                                           relu=False)
            else:
                concat = tf.concat(axis=3, values=[conv_0, conv_1, conv_2], name='concat')
                proj = input

            conv = self.conv._original(self, concat, 1, 1, c_out, 1, 1, name='out/conv', relu=False)
            if name == 'conv5_4':
                conv = self.bn_scale_combo._original(self, conv, c_in=c_out, name='out', relu=False)
            conv = self.add._original(self, [conv, proj], name='sum')
        return  conv

    @layer
    def pva_inception_res_block(self, input, name, name_prefix = 'conv4_', type = 'a'):
        """build inception block"""
        node = input
        if type == 'a':
            c_ins = (128, 256, 256, 256, 256, )
        else:
            c_ins = (256, 384, 384, 384, 384, )
        for i in range(1, 5):
            node = self.pva_inception_res_stack._original(self, node, c_in = c_ins[i-1],
                                                          name = name_prefix + str(i), block_start=(i==1), type=type)
        return node

    @layer
    def scale(self, input, c_in, name):
        with tf.variable_scope(name) as scope:

            alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=True,
                                    regularizer=self.l2_regularizer(0.00001))
            beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0), trainable=True,
                                   regularizer=self.l2_regularizer(0.00001))
            return tf.add(tf.multiply(input, alpha), beta)






    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)



    def build_loss(self, ohem=False):
        ############# RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        # ignore_label(-1)
        fg_keep = tf.equal(rpn_label, 1)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep), [-1, 2]) # shape (N, 2)
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep), [-1])
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

        # box loss
        rpn_bbox_pred = self.get_output('rpn_bbox_pred') # shape (1, H, W, Ax4)
        rpn_bbox_targets = self.get_output('rpn-data')[1]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[2]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[3]
        rpn_bbox_pred = tf.reshape(tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep), [-1, 4]) # shape (N, 4)
        rpn_bbox_targets = tf.reshape(tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep), [-1, 4])
        rpn_bbox_inside_weights = tf.reshape(tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep), [-1, 4])
        rpn_bbox_outside_weights = tf.reshape(tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep), [-1, 4])

        rpn_loss_box_n = tf.reduce_sum(self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), axis=[1])

        # rpn_loss_n = tf.reshape(rpn_cross_entropy_n + rpn_loss_box_n * 5, [-1])

        if ohem:
            # k = tf.minimum(tf.shape(rpn_cross_entropy_n)[0] / 2, 300)
            # # k = tf.shape(rpn_loss_n)[0] / 2
            # rpn_loss_n, top_k_indices = tf.nn.top_k(rpn_cross_entropy_n, k=k, sorted=False)
            # rpn_cross_entropy_n = tf.gather(rpn_cross_entropy_n, top_k_indices)
            # rpn_loss_box_n = tf.gather(rpn_loss_box_n, top_k_indices)

            # strategy: keeps all the positive samples
            fg_ = tf.equal(rpn_label, 1)
            bg_ = tf.equal(rpn_label, 0)
            pos_inds = tf.where(fg_)
            neg_inds = tf.where(bg_)
            rpn_cross_entropy_n_pos = tf.reshape(tf.gather(rpn_cross_entropy_n, pos_inds), [-1])
            rpn_cross_entropy_n_neg = tf.reshape(tf.gather(rpn_cross_entropy_n, neg_inds), [-1])
            top_k = tf.cast(tf.minimum(tf.shape(rpn_cross_entropy_n_neg)[0], 300), tf.int32)
            rpn_cross_entropy_n_neg, _ = tf.nn.top_k(rpn_cross_entropy_n_neg, k=top_k)
            rpn_cross_entropy = tf.reduce_sum(rpn_cross_entropy_n_neg) / (tf.reduce_sum(tf.cast(bg_, tf.float32)) + 1.0) \
                                + tf.reduce_sum(rpn_cross_entropy_n_pos) / (tf.reduce_sum(tf.cast(fg_, tf.float32)) + 1.0)

            rpn_loss_box_n = tf.reshape(tf.gather(rpn_loss_box_n, pos_inds), [-1])
            # rpn_cross_entropy_n = tf.concat(0, (rpn_cross_entropy_n_pos, rpn_cross_entropy_n_neg))

        # rpn_loss_box = 1 * tf.reduce_mean(rpn_loss_box_n)
        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1.0)

        ############# R-CNN
        # classification loss
        cls_score = self.get_output('cls_score') # (R, C+1)
        label = tf.reshape(self.get_output('roi-data')[1], [-1]) # (R)
        cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)

        # bounding box regression L1 loss
        bbox_pred = self.get_output('bbox_pred') # (R, (C+1)x4)
        bbox_targets = self.get_output('roi-data')[2] # (R, (C+1)x4)
        # each element is {0, 1}, represents background (0), objects (1)
        bbox_inside_weights = self.get_output('roi-data')[3] # (R, (C+1)x4)
        bbox_outside_weights = self.get_output('roi-data')[4] # (R, (C+1)x4)

        loss_box_n = tf.reduce_sum( \
            bbox_outside_weights * self.smooth_l1_dist(bbox_inside_weights * (bbox_pred - bbox_targets)), \
            axis=[1])

        loss_n = loss_box_n + cross_entropy_n
        loss_n = tf.reshape(loss_n, [-1])

        # if ohem:
        #     # top_k = 100
        #     top_k = tf.minimum(tf.shape(loss_n)[0] / 2, 500)
        #     loss_n, top_k_indices = tf.nn.top_k(loss_n, k=top_k, sorted=False)
        #     loss_box_n = tf.gather(loss_box_n, top_k_indices)
        #     cross_entropy_n = tf.gather(cross_entropy_n, top_k_indices)

        loss_box = tf.reduce_mean(loss_box_n)
        cross_entropy = tf.reduce_mean(cross_entropy_n)

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss

        return loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box