import unittest
import tensorflow as tf
from transformer import MultiheadAttention


class TestTransformer(unittest.TestCase):
    def test_multihead_attention(self):
        head_num = 2
        batch_size = 3
        max_q_len = 5
        max_k_len = 7
        query_dim = 22
        key_dim = 26
        value_dim = 34
        layer = MultiheadAttention(head_num, k_dim=key_dim, v_dim=value_dim)
        q = tf.ones(shape=[batch_size, max_q_len, query_dim])
        k = tf.ones(shape=[batch_size, max_k_len, query_dim])
        v = tf.ones(shape=[batch_size, max_k_len, query_dim])
        output = layer([q, k, v])
        self.assertEqual(output.shape, [batch_size, max_q_len, query_dim])

        with tf.Graph().as_default():
            layer = MultiheadAttention(head_num, k_dim=key_dim, v_dim=value_dim)
            q = tf.ones(shape=[batch_size, max_q_len, query_dim])
            k = tf.ones(shape=[batch_size, max_k_len, query_dim])
            v = tf.ones(shape=[batch_size, max_k_len, query_dim])
            output = layer([q, k, v])
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                result = sess.run(output)
                self.assertEqual(result.shape, (batch_size, max_q_len, query_dim))


    def test_split_head(self):
        head_num = 2
        batch_size = 3
        dim = 10
        max_len = 7
        layer = MultiheadAttention(head_num=head_num)
        result = layer._split_head(tf.ones(shape=[batch_size, max_len, dim]))
        self.assertEqual(result.shape, [batch_size, head_num, max_len, int(dim / head_num)])
