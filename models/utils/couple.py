import tensorflow as tf
from config import ModelBasic


def in_block(inputs, size, name='in_block', use_bias=True):
    out = tf.layers.dense(inputs=inputs,
                          units=size,
                          use_bias=use_bias,
                          name=name + '_fc')
    out = tf.nn.relu(out, name=name + '_relu')
    return out


def mid_block(inputs, size, hidden_layers, name='mid_block', use_bias=False):
    for i in range(hidden_layers):
        out = tf.layers.dense(inputs=inputs,
                              units=size,
                              use_bias=use_bias,
                              name=name + '_fc_' + str(i))
        out = tf.nn.relu(out, name=name + '_relu_' + str(i))
    return out


def out_block(inputs, size, name='out_block', use_bias=True):
    out = tf.layers.dense(inputs=inputs,
                          units=size,
                          use_bias=use_bias,
                          name=name + '_fc_')
    return out


def couple(inputs, batch_size, in_out_dim, mid_dim, hidden_dim, name='couple', mask_config=0, reverse=False):
    """couple_layer infer.
    Args:
        x: input tensor.
        reverse: True in inference mode, False in sampling mode.
    Returns:
        transformed tensor.
    """
    B = batch_size
    # B = inputs.get_shape().as_list()[0]
    # [B, W] = tf.shape(inputs)
    h = tf.reshape(inputs, [B, ModelBasic.in_out_dim//2, 2], name=name+'_reshape_0')
    if mask_config:
        on, off = h[:, :, 0], h[:, :, 1]
    else:
        off, on = h[:, :, 0], h[:, :, 1]

    print('off.shape', off.shape)
    off_ = in_block(off, mid_dim, name=name+'_in_block')
    off_ = mid_block(off_, mid_dim, hidden_dim, name=name+'_mid_block')
    shift = out_block(off_, in_out_dim//2, name=name+'_out_block')

    if reverse:
        on = on - shift
    else:
        on = on + shift

    if mask_config:
        x = tf.stack((on, off), axis=2, name=name + '_stack')
    else:
        x = tf.stack((off, on), axis=2, name=name + '_stack')
    return tf.reshape(x, [B, -1], name=name+'_reshape_1')


def couple_layer(inputs, batch_size, couple_layers, in_out_dim, mid_dim, hidden_dim, mask_config, name='couple_layer', generate=True):
    layer_index = list(range(couple_layers))
    if generate:
        reverse = False
        reuse = True
        # layer_index = range(couple_layers)
    else:
        reverse = True
        reuse = False
        # layer_index = reversed(range(couple_layers))
        layer_index.reverse()
    #
    # print()
    # print(name, 'reuse is ', reuse)
    # print('mask_config is ', mask_config)
    # for i in layer_index:
    #     print('layer_index is ', i)
    # print()
    # with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope(name):
        for i in layer_index:
            # print('couple_layer_', i)
            # print('inputs.shape:', inputs.shape)
            inputs = couple(inputs, batch_size, in_out_dim, mid_dim, hidden_dim, name='couple' + str(i), mask_config=(mask_config+i)%2, reverse=reverse)

    return inputs