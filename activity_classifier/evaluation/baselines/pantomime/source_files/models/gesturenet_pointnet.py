"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module

def placeholder_inputs(batch_size, num_frame, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_frame, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_pointnet_module(point_cloud, is_training, reuse_weights, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                        padding='VALID', stride=[1,1],
                        reuse_weights=reuse_weights,
                        bn=True, is_training=is_training,
                        scope='pointnet_conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        reuse_weights=reuse_weights,
                        bn=True, is_training=is_training,
                        scope='pointnet_conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                        padding='VALID', stride=[1,1],
                        reuse_weights=reuse_weights,
                        bn=True, is_training=is_training,
                        scope='pointnet_conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                        padding='VALID', stride=[1,1],
                        reuse_weights=reuse_weights,
                        bn=True, is_training=is_training,
                        scope='pointnet_conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                            reuse_weights=reuse_weights,
                            padding='VALID', scope='pointnet_maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                reuse_weights=reuse_weights,
                                scope='pointnet_fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                        reuse_weights=reuse_weights,  
                        scope='pointnet_dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                reuse_weights=reuse_weights,
                                scope='pointnet_fc2', bn_decay=bn_decay)
    return net

def get_model(point_cloud, is_training, num_classes, bn_decay=None):
    """ Classification GestureNet, input is BxFxNx3, output Bx11 """
    batch_size = point_cloud.get_shape()[0].value
    num_frame = point_cloud.get_shape()[1].value
    num_point = point_cloud.get_shape()[2].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    spatial_feature_extraction_output = None
    for frame_index in range(num_frame):
        point_net_input = tf.slice(l0_xyz, [0, frame_index, 0, 0], [batch_size, 1, num_point, 3])
        point_net_input = tf.reshape(point_net_input, [-1])
        point_net_input = tf.reshape(point_net_input, [batch_size, num_point, 3])
        net = get_pointnet_module(point_net_input, is_training, frame_index > 0, bn_decay)
        if spatial_feature_extraction_output is None:
            spatial_feature_extraction_output = tf.expand_dims(net, 0)
        else:
            spatial_feature_extraction_output = tf.concat([spatial_feature_extraction_output, tf.expand_dims(net, 0)], 0)
    net = tf.transpose(spatial_feature_extraction_output, [1, 0, 2])
    net = tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)(net, scope='lstm1', training=is_training)
    net = tf.keras.layers.LSTM(32, dropout=0.4, recurrent_dropout=0.4)(net, scope='lstm2', training=is_training)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
