import logging
from operator import mul
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from functools import reduce
from tensorflow.contrib import graph_editor as ge
import os
from config import TrainBasic

"""Standard logistic distribution.
"""


def softplus(x):
    return tf.log(1. + tf.exp(x))


class StandardLogistic:
    def __init__(self):
        super(StandardLogistic, self).__init__()

    def log_prob(self, x):
        """Computes data log-likelihood.

        Args:
            x: input tensor.
        Returns:
            log-likelihood.
        """
        return -(tf.nn.softplus(x) + tf.nn.softplus(-x))

    @staticmethod
    def sample(size):
        """Samples from the distribution.

        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = np.random.uniform(0., 1., size)
        return np.log(z) - np.log(1. - z)


def check_restore_params(saver, sess, run_name, corpus_name=TrainBasic.dataset):
    """
    if checkpoints exits in the give path, then restore the parameters and return True

    :param saver: tf.train.Saver
    :param sess: tf.Session
    :param run_name: directory name of the checkpoints
    :param corpus_name: corpus name
    :return: boolean, return true if checkpoints found else False
    """
    directory = os.path.join(TrainBasic.checkpoint_dir, corpus_name, run_name, '')
    print('')
    print('')
    print(directory)
    print('')
    print('')
    checkpoint = tf.train.get_checkpoint_state(directory)
    if checkpoint and checkpoint.model_checkpoint_path:
        logging.info('Restoring checkpoints...')
        saver.restore(sess, checkpoint.model_checkpoint_path)
        return True
    else:
        logging.info('No checkpoint found, use initialized parameters.')
        return False


def check_op_name(regex='Block./layer.*/concat.*'):
    ops = tf.get_default_graph().get_operations()
    concat_ops_list = ge.filter_ops_from_regex(ops, regex)
    for op in ops:
        if op not in concat_ops_list and op.outputs:
            print(op)
            logging.info(' out: {0}'.format(op.outputs[0]))


def check_param_num():
    logging.info('checking parameter number...')
    num = 0
    param_dict = {}
    for var in tf.trainable_variables():
        shape = var.get_shape()
        cur_num = reduce(mul, [dim.value for dim in shape], 1)
        param_dict[var.name] = cur_num
        num += cur_num
    logging.info('total parameter: {0}'.format(num))
    logging.info(param_dict)


def basic_config():
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('max_colwidth', 100)
    np.set_printoptions(precision=3, edgeitems=8, suppress=True, linewidth=1000)
