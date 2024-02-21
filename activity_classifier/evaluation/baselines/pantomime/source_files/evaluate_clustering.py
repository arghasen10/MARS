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
parser.add_argument('--num_class', type=int, default=21, help='Number of training classes. [default: 21]')
parser.add_argument('--dataset', default='/scratch/work/salamid1/JDataset/8frames/1024points', help='Dataset path. [default: /scratch/work/salamid1/JDataset/8frames/1024points]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
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

TEST_DATASET = gesturenet_h5_dataset.GestureNetH5Dataset('{}/{}_files.txt'.format(DATASET, SET_TO_EVALUATE), nframes=NUM_FRAME, batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
    if GPU_INDEX < 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = '/cpu:0'
    else:
        device = '/gpu:' + str(GPU_INDEX)
    with tf.device(device):
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
           'final_latent_space': end_points['final_latent_space'],
           }

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

    total_y_true = []
    total_latent_space = None
    total_y_score = None
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batch_latent_space = np.zeros((BATCH_SIZE, 256))
        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            # Shuffle point order to achieve different farthest samplings
            shuffled_indices = np.arange(NUM_POINT)
            np.random.shuffle(shuffled_indices)
            if FLAGS.normal:
                rotated_data = provider.rotate_point_cloud_by_angle_with_normal(cur_batch_data[:,:, shuffled_indices, :],
                    vote_idx/float(num_votes) * np.pi * 2)
            else:
                rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:,:, shuffled_indices, :],
                    vote_idx/float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training,
                         tf.keras.backend.learning_phase(): 1,}
            # loss_val, pred_val, l2_indices, l2_points_before_pooling, l2_xyz = sess.run([ops['loss'], ops['pred'], ops['l2_indices'], ops['l2_points_before_pooling'], ops['l2_xyz']], feed_dict=feed_dict)
            loss_val, pred_val, latent_space = sess.run([ops['loss'], ops['pred'], ops['final_latent_space']], feed_dict=feed_dict)

            # total_l2_indices.append(l2_indices)
            # total_l2_points_before_pooling.append(l2_points_before_pooling)
            # total_l2_xyz.append(l2_xyz)
            batch_latent_space += latent_space
            batch_pred_sum += pred_val
        if total_latent_space is None:
            total_latent_space = batch_latent_space[0:bsize]
        else:
            total_latent_space = np.vstack((total_latent_space, batch_latent_space[0:bsize]))
        if total_y_score is None:
            total_y_score = batch_pred_sum[0:bsize]
        else:
            total_y_score = np.vstack((total_y_score, batch_pred_sum[0:bsize]))
        total_y_true = np.concatenate((total_y_true, batch_label[0:bsize]))
        loss_sum += loss_val
        batch_idx += 1

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))

    with open('{}/scores_{}.pkl'.format(DATASET, FLAGS.model), 'wb') as handle:
        pickle.dump({
            'total_y_true': total_y_true,
            'SHAPE_NAMES': SHAPE_NAMES,
            'total_y_score': total_y_score,
            'total_latent_space': total_latent_space
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log_string('Scores have been saved to the dataset directory!')


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
