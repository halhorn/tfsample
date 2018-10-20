import tensorflow as tf
from typing import cast, Optional, Union, Sequence

PAD = 0.0


class MultiheadAttention(tf.keras.layers.Layer):
    '''
    MultiheadAttention for Transformer
    see:
      - https://arxiv.org/pdf/1706.03762.pdf
      - https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    '''
    def __init__(
            self,
            head_num: int,
            k_dim: Optional[int]=None,
            v_dim: Optional[int]=None,
            o_dim: Optional[int]=None,
            keep_prob: Union[tf.Tensor, float]=1.0,
            masks_future: bool=False,
            **kwargs
    ) -> None:
        '''
        コンストラクタです。

        :param head_num: ヘッダの数
        :param k_dim: 内部での key の次元
            head_num の整数倍である必要があります。
            デフォルトで入力の key の次元と同じになります。
        :param v_dim: 内部での value の次元
            head_num の整数倍である必要があります。
            デフォルトで入力の value の次元と同じになります。
        :param o_dim: 出力の次元
            デフォルトで入力の query の次元と同じになります。
        :param keep_prob: attention weights のドロップアウトの保持率
        :param masks_future: 未来の情報をマスクするか
            Decoder 部分などでself-attention をする際に
            自身の未来の情報をマスクする場合には Trueにします。
        '''
        super(MultiheadAttention, self).__init__(**kwargs)
        self.head_num = head_num
        self.o_dim = o_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.keep_prob = keep_prob
        self.masks_future = masks_future

    def build(self, input_shape: Sequence[tf.TensorShape]) -> None:
        if any(shape[-1].value is None for shape in input_shape):
            raise ValueError('The last dimension of the inputs should be defined.')

        q_shape, k_shape, v_shape = tuple(input_shape)
        self.k_dim = self.k_dim or k_shape[-1].value
        self.v_dim = self.v_dim or v_shape[-1].value
        self.o_dim = self.o_dim or q_shape[-1].value

        assert self.k_dim % self.head_num == 0, 'fail: k_dim {} % head_num {} == 0'.format(self.k_dim, self.head_num)
        assert self.v_dim % self.head_num == 0, 'fail: v_dim {} % head_num {} == 0'.format(self.v_dim, self.head_num)

        self.w_q = self.add_variable('w_q', [q_shape[-1].value, self.k_dim])
        self.w_k = self.add_variable('w_k', [k_shape[-1].value, self.k_dim])
        self.w_v = self.add_variable('w_v', [v_shape[-1].value, self.v_dim])
        self.w_o = self.add_variable('w_o', [self.v_dim, self.o_dim])

    def call(self, input: Sequence[tf.Tensor]) -> tf.Tensor:
        '''
        MultiheadAttention を実行します。
        :param input: query, key, value の Tensor のリスト
        '''
        q, k, v = tuple(input)
        q = tf.tensordot(q, self.w_q, axes=1)  # [batch_size, max_q_len, k_dim]
        k = tf.tensordot(k, self.w_k, axes=1)  # [batch_size, max_k_len, k_dim]
        v = tf.tensordot(v, self.w_v, axes=1)  # [batch_size, max_k_len, v_dim]

        head_q = self._split_head(q)  # [batch_size, head_num, max_q_len, k_dim/head_num]
        head_k = self._split_head(k)  # [batch_size, head_num, max_k_len, k_dim/head_num]
        head_v = self._split_head(v)  # [batch_size, head_num, max_k_len, v_dim/head_num]

        k_dim_per_head = cast(int, self.k_dim) // self.head_num
        head_qk = tf.matmul(head_q, head_k, transpose_b=True) / k_dim_per_head ** 0.5
        mask = tf.equal(head_qk, PAD)
        if self.masks_future:
            mask = tf.linalg.band_part(mask, -1, 0)  # 下三角行列に
        head_qk = self._mask(head_qk, mask, head_qk.dtype.min)  # softmax で exp にかけられるため
        # [batch_size, head_num, max_q_len, max_k_len]
        attention_weight = tf.nn.dropout(tf.nn.softmax(head_qk), keep_prob=self.keep_prob)
        attention_weight = self._mask(attention_weight, mask, PAD)
        # [batch_size, head_num, max_q_len, v_dim/head_num]
        attention = tf.matmul(attention_weight, head_v)
        # [batch_size, max_q_len, v_dim]
        concatenated_attention = self._concat_head(attention)
        # [batch_size, max_q_len, o_dim]
        return tf.tensordot(concatenated_attention, self.w_o, axes=1)

    def _split_head(self, input: tf.Tensor) -> tf.Tensor:
        batch_size, max_len, _ = tf.unstack(tf.shape(input))
        reshaped = tf.reshape(input, [
            batch_size,
            max_len,
            self.head_num,
            -1,  # k_dim / head_num
        ])
        # [batch_size, head_num, max_len, dim/head_num]
        return tf.transpose(reshaped, [0, 2, 1, 3])

    def _concat_head(self, input: tf.Tensor) -> tf.Tensor:
        batch_size, _, max_len, _ = tf.unstack(tf.shape(input))
        return tf.reshape(tf.transpose(input, [0, 2, 1, 3]), [batch_size, max_len, -1])

    def _mask(self, tensor: tf.Tensor, mask: tf.Tensor, mask_value: float) -> tf.Tensor:
        mask_value_tensor = tf.ones_like(tensor) * mask_value
        return tf.where(mask, mask_value_tensor, tensor)
