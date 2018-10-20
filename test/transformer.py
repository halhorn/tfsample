import unittest
import tensorflow as tf
import numpy as np
from transformer import MultiheadAttention

tf.enable_eager_execution()


class TestTransformer(unittest.TestCase):
    def test_multihead_attention(self):
        head_num = 2
        batch_size = 3
        max_q_len = 5
        max_k_len = 7
        key_dim = 26
        value_dim = 34
        out_dim = 19
        in_key_dim = 23
        in_value_dim = 29
        in_query_dim = 31
        layer = MultiheadAttention(head_num, o_dim=out_dim, k_dim=key_dim, v_dim=value_dim, keep_prob=0.7)
        q = tf.ones(shape=[batch_size, max_q_len, in_query_dim])
        k = tf.ones(shape=[batch_size, max_k_len, in_key_dim])
        v = tf.ones(shape=[batch_size, max_k_len, in_value_dim])
        output = layer([q, k, v])
        self.assertEqual(output.shape, [batch_size, max_q_len, out_dim])

        with tf.Graph().as_default():
            layer = MultiheadAttention(head_num, o_dim=out_dim)
            q = tf.placeholder(shape=[None, None, in_query_dim * head_num], dtype=tf.float32)
            output = layer([q, q, q])
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                result = sess.run(output, feed_dict={
                    q: np.ones(shape=[batch_size, max_q_len, in_query_dim * head_num]),
                })
                self.assertEqual(result.shape, (batch_size, max_q_len, out_dim))

    def test_multihead_attention_pad(self):
        head_num = 3
        key_dim = 6
        value_dim = 6
        q = tf.constant([[[1, 1, 1], [0, 0, 0]]], dtype=tf.float32)
        k = tf.constant([[[1, 1, 1], [0, 0, 0]]], dtype=tf.float32)
        layer = MultiheadAttention(head_num, k_dim=key_dim, v_dim=value_dim)
        output = layer([q, k, k])
        self.assertEqual(output.numpy()[0, 1, :].tolist(), [0.0, 0.0, 0.0])

    def test_multihead_attention_masks_future(self):
        batch_size = 2
        head_num = 3
        dim = 6
        max_q_len = 7
        q = tf.ones(shape=[batch_size, max_q_len, dim])
        layer = MultiheadAttention(head_num, masks_future=True)
        output = layer([q, q, q])
        self.assertEqual(output.shape, [batch_size, max_q_len, dim])

    def test_split_head(self):
        head_num = 2
        batch_size = 3
        dim = 10
        max_len = 7
        layer = MultiheadAttention(head_num=head_num)
        result = layer._split_head(tf.ones(shape=[batch_size, max_len, dim]))
        self.assertEqual(result.shape, [batch_size, head_num, max_len, int(dim / head_num)])
