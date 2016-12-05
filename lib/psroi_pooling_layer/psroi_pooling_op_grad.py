import tensorflow as tf
from tensorflow.python.framework import ops
import psroi_pooling_op
import pdb


@tf.RegisterShape("PSROIPool")
def _psroi_pool_shape(op):
  """Shape function for the PSROIPool op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  channels = dims_data[3]
  dims_rois = op.inputs[1].get_shape().as_list()
  num_rois = dims_rois[0]
  output_dim = op.get_attr('output_dim')
  group_size  = op.get_attr('group_size')
  pooled_height = group_size
  pooled_width = group_size

  output_shape = tf.TensorShape([num_rois, pooled_height, pooled_width, output_dim])
  return [output_shape, output_shape]

@ops.RegisterGradient("PSROIPool")
def _psroi_pool_grad(op, grad, _):
  """The gradients for `PSROI_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  
  data = op.inputs[0]
  rois = op.inputs[1]
  mapping_channel = op.outputs[1]
  spatial_scale = op.get_attr('spatial_scale')

  # compute gradient
  #data_grad = psroi_pooling_op.psroi_pool_grad(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)
  data_grad = psroi_pooling_op.psroi_pool_grad(data, rois, mapping_channel, grad, spatial_scale)  

  return [data_grad, None]  # List of one Tensor, since we have one input

