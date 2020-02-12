from config import ModelBasic
import tensorflow as tf
import numpy as np
from models.utils.couple import couple_layer


class Nice:
    def __init__(self,
                 prior,
                 couple_layers = ModelBasic.couple_layers,
                 in_out_dim = ModelBasic.in_out_dim,
                 mid_dim = ModelBasic.mid_dim,
                 hidden_dim=ModelBasic.hidden_dim,
                 mask_config=ModelBasic.mask_config,
                 training=False):
        self._training = training
        # self.in_out_dim = ModelBasic.in_out_dim
        # self.mid_dim = ModelBasic.mid_dim
        # self.hidden = ModelBasic.hidden_dim
        self.prior = prior
        self.couple_layers = couple_layers
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden_dim
        self.mask_config = mask_config

    def scaling(self, z, generate=False):
        reuse = generate
        with tf.variable_scope('ScaleLayer', reuse=reuse):
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
        x = couple_layer(x, self.couple_layers, self.mask_config, name='couple_layer', generate=True)
        return x

    def inv_generate(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        z = couple_layer(x, self.couple_layers, self.mask_config, name='couple_layer', generate=False)
        z = self.scaling(z, generate=False)
        return z

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.inv_generate(x)
        log_ll = tf.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def infer(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)