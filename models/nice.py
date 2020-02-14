from config import ModelBasic, TrainBasic
from utils.basis_util import StandardLogistic
import tensorflow as tf
import numpy as np
from models.utils.couple import couple_layer


class Nice:
    def __init__(self,
                 prior=StandardLogistic(),
                 couple_layers = ModelBasic.couple_layers,
                 in_out_dim = ModelBasic.in_out_dim,
                 mid_dim = ModelBasic.mid_dim,
                 hidden_dim=ModelBasic.hidden_dim,
                 mask_config=ModelBasic.mask_config,
                 batch_size=TrainBasic.batch_size,
                 training=False):
        self._training = training
        # self.in_out_dim = ModelBasic.in_out_dim
        # self.mid_dim = ModelBasic.mid_dim
        # self.hidden = ModelBasic.hidden_dim
        self.batch_size = batch_size
        self.prior = prior
        self.couple_layers = couple_layers
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden_dim = hidden_dim
        self.mask_config = mask_config

    def scaling(self, z, generate=False):
        reuse = generate
        # with tf.variable_scope('ScaleLayer', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('ScaleLayer'):
            self.scale = tf.get_variable('scale',
                                         [1, self.in_out_dim],
                                         # initializer=np.zeros([1, self.in_out_dim],
                                         initializer=tf.constant_initializer(0.0))
            log_det_J = tf.reduce_sum(self.scale)
            if generate:
                z = z * tf.exp(-self.scale)
            else:
                z = z * tf.exp(self.scale)
            return z, log_det_J

    def generate(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, generate=True)
        x = couple_layer(x, self.batch_size, self.couple_layers, self.in_out_dim, self.mid_dim, self.hidden_dim, self.mask_config, name='couple_layer', generate=True)
        return x

    def inv_generate(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        h = couple_layer(x, self.batch_size, self.couple_layers, self.in_out_dim, self.mid_dim, self.hidden_dim, self.mask_config, name='couple_layer', generate=False)
        return self.scaling(h, generate=False)

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.inv_generate(x)
        log_ll = tf.reduce_sum(self.prior.log_prob(z), axis=1)
        return log_ll + log_det_J

    # def sample(self, size):
    #     """Generates samples.
    #
    #     Args:
    #         size: number of samples to generate.
    #     Returns:
    #         samples from the data space X.
    #     """
    #     z = self.prior.sample((size, self.in_out_dim))
    #     return self.generate(z)

    def infer(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)