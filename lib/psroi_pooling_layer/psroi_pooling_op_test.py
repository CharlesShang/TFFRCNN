import tensorflow as tf
import numpy as np
import psroi_pooling_op
import psroi_pooling_op_grad
import pdb

pdb.set_trace()

rois = tf.convert_to_tensor([ [0, 0, 0, 4, 4]], dtype=tf.float32)
hh=tf.convert_to_tensor(np.random.rand(1,5,5,25),dtype=tf.float32)
[y2, channels] = psroi_pooling_op.psroi_pool(hh, rois, output_dim=1, group_size=5, spatial_scale=1.0)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print( sess.run(hh))
print( sess.run(y2))
pdb.set_trace()
