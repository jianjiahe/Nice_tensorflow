import tensorflow as tf


def in_block(inputs, size, name='in_block', use_bias=True):
    out = tf.layers.dense(inputs,
                          units=size,
                          use_bias=use_bias)
    out = tf.nn.relu(out)
    return out

def mid_block(inputs, size, name='mid_block', use_bias=True):
    out = tf.layers.dense(inputs,
                          units=size,
                          use_bias=use_bias)
    out = tf.nn.relu(out)



def pre_net(inputs, training=False, scope='pre_net', use_bn=False):
    out = inputs if not use_bn else tf.layers.batch_normalization(inputs, axis=-1, training=training)
    layers_sizes = ModelBasic.prenet_depths
    with tf.variable_scope(scope):
        for size in layers_sizes:
            out = tf.layers.dense(
                out,
                units=size,
                use_bias=not use_bn
            )
            if use_bn:
                out = tf.layers.batch_normalization(out, axis=-1, training=training)
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, rate=0.5, training=training)
    return out