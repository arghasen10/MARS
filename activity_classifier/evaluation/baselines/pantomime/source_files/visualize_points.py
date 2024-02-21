'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
from sklearn.metrics import confusion_matrix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import matplotlib
matplotlib.use('agg')
import pylab as plt
import seaborn as sn
import pickle
import pandas as pd
import gesturenet_h5_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='gesturenet', help='Model name. [default: gesturenet]')
parser.add_argument('--num_class', type=int, default=10, help='Number of classes. [default: 21]')
parser.add_argument('--dataset', default='/scratch/work/salamid1/JDataset/8frames/1024points', help='Dataset path. [default: /scratch/work/salamid1/JDataset/8frames/1024points]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=512, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--num_frame', type=int, default=10, help='Frame Number [default: 10]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('--validation_or_test', default='test', help='Which set should be evaluated [test]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frame
DATASET = FLAGS.dataset
MODEL_PATH = FLAGS.model_path
SET_TO_EVALUATE = FLAGS.validation_or_test
GPU_INDEX = FLAGS.gpu
NUM_CLASSES = FLAGS.num_class
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, '{}/gesture_names.txt'.format(DATASET)))]

HOSTNAME = socket.gethostname()

TRAIN_DATASET = gesturenet_h5_dataset.GestureNetH5Dataset('{}/train_files.txt'.format(DATASET), nframes=NUM_FRAME, batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
TEST_DATASET = gesturenet_h5_dataset.GestureNetH5Dataset('{}/{}_files.txt'.format(DATASET, SET_TO_EVALUATE), nframes=NUM_FRAME, batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_FRAME, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, num_classes=NUM_CLASSES)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    log_string("The model restore process started.")
    sys.stdout.flush()
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")
    sys.stdout.flush()

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss,
           'l3_points': end_points['l3_points'],
           # 'l2_points_before_pooling': end_points['l2_points_before_pooling'],
           # 'l2_indices': end_points['l2_indices']
           }
    for i in range(NUM_FRAME):
       ops['l3_points_{}'.format(i)] = end_points['l3_points_{}'.format(i)]

    eval_one_epoch(sess, ops, num_votes)

def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_FRAME,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_y_pred = []
    total_y_true = []
    total_y_score = None
    # total_l2_indices = []
    # total_l2_points_before_pooling = []
    # total_l2_xyz = []
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        start_idx = 0
        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for frame in [0, 3, 4, 7]:
            #-------------------------------------------------------------------
            # get critical points
            #-------------------------------------------------------------------
            no_influence_position = cur_batch_data[start_idx, frame, 0, :].copy()
            global_feature_list = []
            orgin_data = cur_batch_data[start_idx, frame, :, :].copy()

            for change_point in range(NUM_POINT):
                cur_batch_data[start_idx, frame, change_point, :] = no_influence_position.copy()

            for change_point in range(NUM_POINT):
                cur_batch_data[start_idx, frame, change_point, :] = orgin_data[change_point, :].copy()

                for vote_idx in range(num_votes):
                    if FLAGS.normal:
                        rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:,:, :, :],
                            vote_idx/float(num_votes) * np.pi * 2)
                    else:
                        rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:,:, :, :],
                            vote_idx/float(num_votes) * np.pi * 2)
                    feed_dict = {ops['pointclouds_pl']: rotated_data,
                                 ops['labels_pl']: cur_batch_label,
                                 ops['is_training_pl']: is_training,
                                 tf.keras.backend.learning_phase(): 1,}
                    global_feature_val = sess.run(ops['l3_points_{}'.format(frame)],
                                            feed_dict=feed_dict)[0, :, :]
                    global_feature_list.append(global_feature_val)
            critical_points = []
            max_feature = np.zeros(global_feature_list[0].size) - 10
            feature_points = np.zeros(global_feature_list[0].size)
            for index in range(len(global_feature_list)):
                top = global_feature_list[index]
                feature_points = np.where(top > max_feature, index, feature_points)
                max_feature = np.where(top > max_feature, top, max_feature)

            for index in feature_points[0]:
                critical_points.append(orgin_data[int(index), :])
            critical_points = list(set([tuple(t) for t in critical_points]))
            print(len(critical_points))


            with open('{}/critical_points/critical_points_{}_{}.pkl'.format(DATASET, batch_idx, frame), 'wb') as handle:
                pickle.dump({
                    'critical_points': critical_points,
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)


            #-------------------------------------------------------------------
            # get upper-bound points
            #-------------------------------------------------------------------
            upper_bound_points = np.empty_like(orgin_data.shape)
            upper_bound_points = orgin_data.copy()
            cur_batch_data[start_idx, frame, :, :] = orgin_data.copy()

            search_step = 0.01
            stand_feature = np.empty_like(global_feature_list[-1].shape)
            max_position = [-5,-5,-5]
            min_position = [5, 5, 5]
            for point_index in range(NUM_POINT):
                max_position = np.maximum(max_position, cur_batch_data[start_idx, frame, point_index,:])
                min_position = np.minimum(min_position, cur_batch_data[start_idx, frame, point_index,:])
            print('max_position', max_position)
            print('min_position', min_position)
            for vote_idx in range(num_votes):
                if FLAGS.normal:
                    rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:,:, :, :],
                        vote_idx/float(num_votes) * np.pi * 2)
                else:
                    rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:,:, :, :],
                        vote_idx/float(num_votes) * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['labels_pl']: cur_batch_label,
                             ops['is_training_pl']: is_training,
                             tf.keras.backend.learning_phase(): 1,}
                global_feature_val = sess.run(ops['l3_points_{}'.format(frame)],
                                        feed_dict=feed_dict)[0, :, :]
                stand_feature = global_feature_val.copy()
                print('stand_feature.shape', stand_feature.shape)

            change_point = 0
            cur_batch_data[start_idx, frame, :, :] = orgin_data.copy()
            for point_index in range(NUM_POINT):
                if not (point_index in feature_points[0]):
                    change_point = point_index
                    break
            print('change_point', change_point)
            x_points = np.linspace(min_position[0], max_position[0], (max_position[0] - min_position[0]) // search_step + 1)
            y_points = np.linspace(min_position[1], max_position[1], (max_position[1] - min_position[1]) // search_step + 1)
            z_points = np.linspace(min_position[2], max_position[2], (max_position[2] - min_position[2]) // search_step + 1)
            total_number_of_points_to_test = len(x_points) * len(y_points) * len(z_points)
            current_point_index_to_test = 0
            for x in x_points:
                for y in y_points:
                    for z in z_points:
                        cur_batch_data[start_idx, frame, change_point, :] = [x, y, z] #+ orgin_position

                        # Aggregating BEG
                        for vote_idx in range(num_votes):
                            if FLAGS.normal:
                                rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:,:, :, :],
                                    vote_idx/float(num_votes) * np.pi * 2)
                            else:
                                rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:,:, :, :],
                                    vote_idx/float(num_votes) * np.pi * 2)
                            feed_dict = {ops['pointclouds_pl']: rotated_data,
                                         ops['labels_pl']: cur_batch_label,
                                         ops['is_training_pl']: is_training,
                                         tf.keras.backend.learning_phase(): 1,}
                            global_feature_val = sess.run(ops['l3_points_{}'.format(frame)],
                                                    feed_dict=feed_dict)[0, :, :]

                            if (global_feature_val <= stand_feature).all():
                                print('({}, {}, {}) was added to the upper bound!'.format(x, y, z))
                                upper_bound_points = np.append(upper_bound_points, np.array([[x,y,z]]), axis=0)
                        current_point_index_to_test += 1
                        if current_point_index_to_test % 1000 == 0:
                            print('Point {}/{} in frame {}'.format(current_point_index_to_test, total_number_of_points_to_test, frame))
                            print('Number of upper bound points: {}'.format(len(upper_bound_points)))
                        sys.stdout.flush()


            with open('{}/upper_bound/upper_bound_points_{}_{}.pkl'.format(DATASET, batch_idx, frame), 'wb') as handle:
                pickle.dump({
                    'upper_bound_points': upper_bound_points,
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)

            cur_batch_data[start_idx, frame, :, :] = orgin_data.copy()
        break


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
