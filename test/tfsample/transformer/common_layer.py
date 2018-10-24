import unittest
import tensorflow as tf
import numpy as np
import itertools
from tfsample.transformer.common_layer import AddPositionalEncoding, nonzero_vector_mask

tf.enable_eager_execution()


class TestAddPositionalEncoding(unittest.TestCase):
    def test_call(self):
        max_length = 2
        batch_size = 3
        depth = 7

        layer = AddPositionalEncoding()
        input = tf.ones(shape=[batch_size, max_length, depth])
        result = layer(input)
        self.assertEqual(result.shape, [batch_size, max_length, depth])
        positional_encoding = (result - input).numpy()

        # PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
        # PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
        for batch, i, pos in itertools.product(range(batch_size), range(depth // 2), range(max_length)):
            self.assertAlmostEqual(
                positional_encoding[batch, pos, i * 2],
                np.sin(pos / 10000 ** (i * 2 / depth)),
                places=6,
            )
            self.assertAlmostEqual(
                positional_encoding[batch, pos, i * 2 + 1],
                np.cos(pos / 10000 ** (i * 2 / depth)),
                places=6,
            )

    def test_call_pad(self):
        inputs = tf.constant([
            [[1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
        ])
        positional_encoding = (AddPositionalEncoding()(inputs) - inputs).numpy()
        self.assertNotEqual(positional_encoding[0, 0, :].sum(), 0)
        self.assertNotEqual(positional_encoding[0, 1, :].sum(), 0)
        self.assertEqual(positional_encoding[0, 2, :].sum(), 0)

    def test_call_graph(self):
        batch_size = 3
        max_length = 5
        depth = 7
        data = np.ones(shape=[batch_size, max_length, depth])

        with tf.Graph().as_default():
            with tf.Session() as sess:
                layer = AddPositionalEncoding()
                input = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
                result_op = layer(input)
                result = sess.run(result_op, feed_dict={
                    input: data,
                })
                self.assertEqual(result.shape, (batch_size, max_length, depth))

        positional_encoding = result - data

        # PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
        # PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
        for batch, i, pos in itertools.product(range(batch_size), range(depth // 2), range(max_length)):
            self.assertAlmostEqual(
                positional_encoding[batch, pos, i * 2],
                np.sin(pos / 10000 ** (i * 2 / depth)),
                places=6,
            )
            self.assertAlmostEqual(
                positional_encoding[batch, pos, i * 2 + 1],
                np.cos(pos / 10000 ** (i * 2 / depth)),
                places=6,
            )


class TestNonzeroVectorMask(unittest.TestCase):
    def test_nonzero_vector_mask(self):
        inputs = tf.constant([[0, 0, 0],
                              [0, 5, -5],
                              [0, 1, 0],
                              [0, 0, 0],
                              [0, 2, 3]])
        expects = [[0, 0, 0],
                   [1, 1, 1],
                   [1, 1, 1],
                   [0, 0, 0],
                   [1, 1, 1]]
        self.assertEqual(nonzero_vector_mask(inputs).numpy().tolist(), expects)
