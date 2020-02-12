from config import ModelBasic
import tensorflow as tf
import numpy as np


class Nice:
    def __init__(self, couple_layers, in_out_dim, mid_dim, hidden_dim, training=False):
        self._training = training
        # self.in_out_dim = ModelBasic.in_out_dim
        # self.mid_dim = ModelBasic.mid_dim
        # self.hidden = ModelBasic.hidden_dim
        self.couple_layers = couple_layers
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden_dim

    def scaling(self, z, reverse=True):
        with tf.variable_scope('ScaleLayer'):
            self.scale = tf.get_variable('scale',
                                         [1, self.in_out_dim],
                                         # initializer=np.zeros([1, self.in_out_dim],
                                         initializer=tf.constant_initializer(0.0))
            log_det_J = tf.reduce_sum(self.scale)
            if reverse:
                z = z * tf.exp(-self.scale)
            else:
                z = z * tf.exp(self.scale)
            return z, log_det_J

    def coupling(self, h, mask_config, reverse=True, reuse=False):
        [B, W] = tf.shape(h)
        h = tf.reshape(h, [B, W//2, 2])
        
        # with tf.variable_scope('coupleLayer',reuse=reuse):





    def generate(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        h, _ = self.scaling(z, reverse=True)
        for i in reversed(range(self.couple_layers)):
            h = self.coupling(h, reverse=True, name='couple_' + str(i))
        x = h
        tf.constant_initializer(0.4)
        return x

    def