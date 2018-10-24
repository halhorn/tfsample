import tensorflow as tf
import math

PAD = 0.0


class AddPositionalEncoding(tf.keras.layers.Layer):
    '''
    入力テンソルに対し、位置の情報を付与して返すレイヤーです。
    see: https://arxiv.org/pdf/1706.03762.pdf

    PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
    PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
    '''
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        fl_type = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))

        depth_counter = tf.range(depth) // 2 * 2  # 0, 0, 2, 2, 4, ...
        depth_matrix = tf.tile(tf.expand_dims(depth_counter, 0), [max_length, 1])
        depth_matrix = tf.pow(10000.0, tf.cast(depth_matrix / depth, fl_type))  # [max_length, depth]

        # cos(x) == sin(x + π/2)
        phase = tf.cast(tf.range(depth) % 2, fl_type) * math.pi / 2  # 0, π/2, 0, π/2, ...
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [max_length, 1])  # [max_length, depth]

        pos_counter = tf.range(max_length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1), [1, depth]), fl_type)  # [max_length, depth]

        positional_encoding = tf.sin(pos_matrix / depth_matrix + phase_matrix)
        # [batch_size, max_length, depth]
        positional_encoding = tf.tile(tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1])

        mask = 1.0 - tf.cast(tf.equal(inputs, PAD), fl_type)  # not equal
        return inputs + positional_encoding * mask


def nonzero_vector_mask(inputs):
    '''
    [[0, 0, 0],
     [0, 5, -5],
     [1, 2, 3]]
    といった任意のランクの行列があるときに、
    以下のように axis=-1 がゼロベクトルで無い部分が1となるマスクを返します。
    [[0, 0, 0],
     [1, 1, 1],
     [1, 1, 1]]
    '''
    zeros_map = tf.equal(inputs, tf.zeros_like(inputs))  # [..., depth]
    all_zero_map = tf.expand_dims(tf.reduce_all(zeros_map, axis=-1), axis=-1)  # [..., 1]
    return tf.ones_like(inputs) - tf.cast(all_zero_map, dtype=inputs.dtype)  # [..., depth]
