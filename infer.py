from utils.InferBase import InferBase
import numpy as np
from config import ModelBasic, TrainBasic
import tensorflow as tf
from models import nice
from utils.basis_util import StandardLogistic
from utils.batch import prepare_data, MiniBatch
import torchvision, torch
import os


class Infer(InferBase):
    def __init__(self,
                 corpus_name,
                 run_name,
                 sample_num=TrainBasic.batch_size,
                 sample_dim=ModelBasic.in_out_dim):
        super(Infer, self).__init__(corpus_name=corpus_name,
                                    run_name=run_name)
        self.sample_num = sample_num
        self.sample_dim = sample_dim

    @staticmethod
    def construct_input(sample_num, sample_dim):
        return StandardLogistic.sample(size=[sample_num, sample_dim])

    def generate(self, epoch, step):
        if not self.has_built:
            self._build_graph()
        with tf.Session() as sess:
            self._graph_init(sess)
            net_input = self.construct_input(self.sample_num, self.sample_dim)
            flow_out = sess.run([self.flow_out], feed_dict={self.inputs: net_input})[0]
            mean = MiniBatch(batch_size=TrainBasic.batch_size, training=True).train_data_mean

            flow_out = prepare_data(flow_out, mean, evaluating=True)

            print()
            print()
            print()
            print('net_input.shape.', net_input.shape)
            print('flow_out.shape.', np.array(flow_out).shape)
            print()
            print()
            print()
            print()

            flow_tensor = torch.from_numpy(flow_out)
            picture_path = './samples/' + TrainBasic.filename + 'epoch%d_' % epoch + 'step%d.png' % step
            print(picture_path)
            if not os.path.exists(os.path.dirname(picture_path)):
                os.makedirs(os.path.dirname(picture_path))
            torchvision.utils.save_image(torchvision.utils.make_grid(flow_tensor), picture_path)

