import os
import numpy as np
import tensorflow as tf
from models.nice import Nice
from config import TrainBasic
from utils.basis_util import check_restore_params


class InferBase:
    def __init__(self,
                 corpus_name,
                 run_name):
        self.corpus_name = corpus_name
        self.run_name = run_name
        self.has_built = False

    def _build_graph(self):
        tf.reset_default_graph()
        with tf.variable_scope('evaluation_data'):
            self.inputs = tf.placeholder(tf.float32, shape=[None, 784])

        self.flow_out = Nice(training=False, batch_size=TrainBasic.batch_size).generate(self.inputs)

        self.writer = tf.summary.FileWriter(os.path.join(TrainBasic.checkpoint_dir, self.corpus_name, self.run_name))

        self.has_built = True

    @staticmethod
    def finalize():
        tf.get_default_graph().finalize()

    def _graph_init(self, sess):
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        check_restore_params(saver, sess, self.run_name, corpus_name=self.corpus_name)

