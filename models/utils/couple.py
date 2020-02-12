import tensorflow as tf


def in_block(inputs, size, name='in_block', use_bias=True):
    out = tf.layers.dense(inputs,
                          units=size,
                          use_bias=use_bias,
                          name=name + '_fc')
    out = tf.nn.relu(out, name=name + '_relu')
    return out


def mid_block(inputs, size, hidden_layers, name='mid_block', use_bias=True):
    for i in range(hidden_layers):
        out = tf.layers.dense(inputs,
                              units=size,
                              use_bias=use_bias,
                              name=name + '_fc_' + str(i))
        out = tf.nn.relu(out, name=name + '_relu_' + str(i))
    return out


def out_block(inputs, size, name='out_block', use_bias=True):
    out = tf.layers.dense(inputs,
                          units=size,
                          use_bias=use_bias,
                          name=name + '_fc_')
    return out


def couple(inputs, name='couple', mask_config=0, reverse=False):
    """couple_layer infer.
    Args:
        x: input tensor.
        reverse: True in inference mode, False in sampling mode.
    Returns:
        transformed tensor.
    """
    [B, W] = tf.shape(inputs)
    h = tf.reshape(inputs, [B, W // 2, 2], name=name+'_reshape_0')
    if mask_config:
        on, off = h[:, :, 0], h[:, :, 1]
    else:
        off, on = h[:, :, 0], h[:, :, 1]
    off_ = in_block(off, name=name)
    off_ = mid_block(off_, name=name)
    shift = out_block(off_, name=name)

    if reverse:
        on = on - shift
    else:
        on = on + shift

    if mask_config:
        x = tf.stack((on, off), dim=2, name=name + '_stack')
    else:
        x = tf.stack((off, on), dim=2, name=name + '_stack')
    return tf.reshape(x, [B, W], name=name+'_reshape_1')