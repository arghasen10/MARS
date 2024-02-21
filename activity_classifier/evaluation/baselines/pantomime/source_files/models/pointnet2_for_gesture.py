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

def get_aggregated_point_net_model(num_frame, l0_xyz, l0_points, batch_size, num_point, is_training, bn_decay):
    point_net_input = tf.reshape(l0_xyz, [-1])
    point_net_input = tf.reshape(point_net_input, [batch_size, num_point * num_frame, 3])
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(point_net_input, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='agg-frame-layer1', use_nchw=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='agg-frame-layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='agg-frame-layer3')
    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='agg-frame-fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='agg-frame-dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='agg-frame-fc2', bn_decay=bn_decay)
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

    output_of_agg_part = get_aggregated_point_net_model(num_frame, l0_xyz, l0_points, batch_size, num_point, is_training, bn_decay)
    # net = tf.concat([output_of_temporal_part, output_of_agg_part], 1)
    # net = tf_util.batch_norm_for_fc(net, is_training=is_training, scope='bn', bn_decay=bn_decay)
    net = tf_util.dropout(output_of_agg_part, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(output_of_agg_part, num_classes, activation_fn=None, scope='global-fc3')

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
