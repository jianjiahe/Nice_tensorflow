import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from config import DataBasic


def dequantize(x):
    '''Dequantize data.

    Add noise sampled from Uniform(0, 1) to each pixel (in [0, 255]).

    Args:
        x: input tensor.
        reverse: True in inference mode, False in training mode.
    Returns:
        dequantized data.
    '''
    noise = np.random.uniform(0, 1, x.shape)
    return (x * 255. + noise) / 256.


def prepare_data(x, mean, training=False):
    """Prepares data .
    In training mode, flatten and dequantize the input.
    In inference mode, reshape tensor into image size.

    Args:
        x: input minibatch.
        mean: center of original dataset.
        training: False if in inference mode, True if in training mode.
    Returns:
        transformed data.
    """
    if training:
        # assert len(list(x.shape)) == 4
        # [B, C, H, W] = list(x.shape)
        # assert [C, H, W] == [1, 28, 28]
        x = dequantize(x)
        # x = x.reshape((B, C * H * W))
        x -= mean
    else:
        # assert len(list(x.shape)) == 2
        # [B, W] = list(x.shape)
        # assert W == 1 * 28 * 28
        # x += mean
        x -= mean
        # x = x.reshape((B, 1, 28, 28))
    return x


class MiniBatch:
    def __init__(self,
                 training,
                 corpus_name='mnist',
                 batch_size=32):
        self.training = training
        self.corpus_name = corpus_name
        self.batch_size = batch_size
        self.train_data = input_data.read_data_sets(DataBasic.dataset_dir, one_hot=True).train
        self.train_data_num = self.train_data.num_examples
        self.train_data_mean = np.mean(self.train_data.images, axis=0)

        self.test_data = input_data.read_data_sets(DataBasic.dataset_dir, one_hot=True).test
        self.test_data_mean = np.mean(self.test_data.images, axis=0)

    def next_batch(self):
        """Prepares data for NICE.

        In training mode, flatten and dequantize the input.
        In inference mode, reshape tensor into image size.

        Args:
            x: input minibatch.
            dataset: name of dataset.
            zca: ZCA whitening transformation matrix.
            mean: center of original dataset.
            reverse: True if in inference mode, False if in training mode.
        Returns:
            transformed data.
        """
        if self.training:
            origin_batch_x, origin_batch_y = self.train_data.next_batch(self.batch_size)
            batch_x = prepare_data(origin_batch_x, self.train_data_mean, training=True)
        else:
            origin_batch_x, origin_batch_y = self.test_data.next_batch(self.batch_size)
            batch_x = prepare_data(origin_batch_x, self.test_data_mean, training=False)
        return batch_x

def main():
    mini_batch = MiniBatch()
    print(mini_batch.train_data_num)
    for i in range(mini_batch.train_data_num//mini_batch.batch_size):
        data = mini_batch.next_batch()
        print(i, ' batch\'s size is :', list(data.shape))
        for index, data_i in enumerate(data):
            print(index, ' data\'s shape is', data_i.shape)
        print('')
