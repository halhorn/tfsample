import unittest
import tensorflow as tf
import numpy as np
from tfsample.transformer.common_layer import AddPositionalEncoding

tf.enable_eager_execution()


class TestAddPositionalEncoding(unittest.TestCase):
    def test_call(self):
        batch_size = 3
        max_length = 5
        depth = 7

        layer = AddPositionalEncoding()
        input = tf.ones(shape=[batch_size, max_length, depth])
        result = layer(input)
        self.assertEqual(result.shape, [batch_size, max_length, depth])

    def test_call_graph(self):
        batch_size = 3
        max_length = 5
        depth = 7

        with tf.Graph().as_default():
            with tf.Session() as sess:
                layer = AddPositionalEncoding()
                input = tf.placeholder(shape=[None, None, depth], dtype=tf.float32)
                result_op = layer(input)
                result = sess.run(result_op, feed_dict={
                    input: np.ones(shape=[batch_size, max_length, depth]),
                })
                self.assertEqual(result.shape, (batch_size, max_length, depth))
