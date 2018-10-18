import tensorflow as tf
from typing import Optional


class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, head_num: int, k_dim: Optional[int]=None, v_dim: Optional[int]=None) -> None:
        super(MultiheadAttention, self).__init__()
        self.head_num = head_num
        self.k_dim = k_dim
        self.v_dim = v_dim

    def build(self, input_shape):
        q_shape, k_shape, v_shape = tuple(input_shape)
        self.q_dim = q_shape[-1]
        self.k_dim = tf.Dimension(self.k_dim or self.q_dim)
        self.v_dim = tf.Dimension(self.v_dim or self.q_dim)
        assert self.k_dim % self.head_num == 0
        assert self.v_dim % self.head_num == 0
        self.w_q = self.add_variable('w_q', [self.q_dim, self.k_dim])
        self.w_k = self.add_variable('w_k', [self.q_dim, self.k_dim])
        self.w_v = self.add_variable('w_v', [self.q_dim, self.v_dim])
        self.w_o = self.add_variable('w_o', [self.v_dim, self.q_dim])

    def call(self, input):
        q, k, v = tuple(input)
        batch_size, max_q_len, _ = q.shape
        q = self._matmul(q, self.w_q)  # [batch_size, max_q_len, k_dim]
        k = self._matmul(k, self.w_k)  # [batch_size, max_k_len, k_dim]
        v = self._matmul(v, self.w_v)  # [batch_size, max_k_len, v_dim]

        head_q = self._split_head(q)  # [batch_size, head_num, max_q_len, k_dim/head_num]
        head_k = self._split_head(k)  # [batch_size, head_num, max_k_len, k_dim/head_num]
        head_v = self._split_head(v)  # [batch_size, head_num, max_k_len, v_dim/head_num]

        head_qk = tf.matmul(head_q, tf.transpose(head_k, [0, 1, 3, 2]))
        d_k = self.k_dim // self.head_num
        # [batch_size, head_num, max_q_len, max_k_len]
        attention_weight = tf.nn.softmax(head_qk / d_k.value ** 0.5)
        # [batch_size, head_num, max_q_len, v_dim/head_num]
        attention = tf.matmul(attention_weight, head_v)
        # [batch_size, max_q_len, v_dim]
        concatenated_attention = tf.reshape(tf.transpose(attention, [0, 2, 1, 3]), [batch_size, max_q_len, -1])
        return self._matmul(concatenated_attention, self.w_o)

    def _split_head(self, input):
        batch_size, max_len, _ = input.shape
        reshaped = tf.reshape(input, [
            batch_size,
            max_len,
            self.head_num,
            -1,  # k_dim / head_num
        ])
        # [batch_size, head_num, max_len, dim/head_num]
        return tf.transpose(reshaped, [0, 2, 1, 3])

    @classmethod
    def _matmul(cls, input, w):
        # input: [a, b, c]
        # w: [c, d]
        # ret: [a, b, d]
        in_shape = input.shape
        output = tf.reshape(input, [-1, in_shape[-1]]) @ w
        return tf.reshape(output, list(in_shape[:-1]) + [-1])
