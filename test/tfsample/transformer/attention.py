import unittest
import tensorflow as tf
import numpy as np
from tfsample.transformer.attention import MultiheadAttention

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

    def test_multihead_attention_fp16(self):
        batch_size = 2
        head_num = 3
        dim = 6
        max_q_len = 7
        q = tf.ones(shape=[batch_size, max_q_len, dim], dtype=tf.float16)
        layer = MultiheadAttention(head_num, masks_future=True)
        output = layer([q, q, q])
        self.assertEqual(output.shape, [batch_size, max_q_len, dim])
        self.assertEqual(output.dtype, tf.float16)

    def test_split_head(self):
        head_num = 2
        batch_size = 3
        dim = 10
        max_len = 7
        layer = MultiheadAttention(head_num=head_num)
        result = layer._split_head(tf.ones(shape=[batch_size, max_len, dim]))
        self.assertEqual(result.shape, [batch_size, head_num, max_len, int(dim / head_num)])

    @unittest.skip('debugging')
    def test_train(self):
        head_num = 8
        dim = 256
        batch_size = 128

        layer = MultiheadAttention(head_num=head_num, k_dim=dim, v_dim=dim, o_dim=3)

        def loss(query, memory, labels):
            result = layer([tf.expand_dims(query, 1), memory, memory])
            result = tf.squeeze(result, [1])
            return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=result)

        def grad(query, memory, labels):
            with tf.GradientTape() as tape:
                loss_val = loss(query, memory, labels)
            return loss_val, tape.gradient(loss_val, layer.weights)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        global_step = tf.train.get_or_create_global_step()

        last_loss = 100
        for _ in range(10000):
            query = tf.random_uniform([batch_size, 3], -1.0, 1.0)
            memory = tf.random_uniform([batch_size, 4, 3], -1.0, 1.0)
            s = tf.reduce_sum(query, -1) + tf.reduce_sum(memory, [1, 2])
            labels = tf.cast(s > 0, tf.int32)
            loss_val, grad_val = grad(query, memory, labels)
            optimizer.apply_gradients(zip(grad_val, layer.weights), global_step)
            if loss_val > last_loss:
                batch_size *= 2
            if int(global_step) % 100 == 0:
                print(global_step.numpy(), loss_val.numpy())
