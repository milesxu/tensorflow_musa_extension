# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MUSA float64 GEMM and elementwise coverage tests."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class DoubleOpsTest(MUSATestCase):
  def _compare(self, op, tensors, rtol=1e-12, atol=1e-12):
    self._compare_cpu_musa_results(
        op, tensors, tf.float64, rtol=rtol, atol=atol
    )

  def test_matmul(self):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    dtype=tf.float64)
    b = tf.constant([[7.0, 8.0], [9.0, 10.0]], dtype=tf.float64)
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_matmul_transpose_a(self):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    dtype=tf.float64)
    b = tf.constant([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                    dtype=tf.float64)
    self._compare(
        lambda x, y: tf.matmul(x, y, transpose_a=True), [a, b]
    )

  def test_batch_matmul(self):
    a = tf.constant(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
         [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
        dtype=tf.float64,
    )
    b = tf.constant(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
         [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
        dtype=tf.float64,
    )
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_batch_matmul_broadcast(self):
    a = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
                    dtype=tf.float64)
    b = tf.constant(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
         [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
        dtype=tf.float64,
    )
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_batch_matmul_strided_batched_large(self):
    a = tf.cast(
        tf.reshape(tf.range(64 * 4 * 48) % 17, [64, 4, 48]), tf.float64
    )
    b = tf.cast(
        tf.reshape(tf.range(64 * 48 * 100) % 13, [64, 48, 100]), tf.float64
    )
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_batch_matmul_strided_broadcast_b(self):
    a = tf.cast(
        tf.reshape(tf.range(64 * 4 * 48) % 17, [64, 4, 48]), tf.float64
    )
    b = tf.cast(tf.reshape(tf.range(48 * 100) % 13, [48, 100]), tf.float64)
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_batch_matmul_strided_broadcast_a(self):
    a = tf.cast(tf.reshape(tf.range(4 * 48) % 17, [4, 48]), tf.float64)
    b = tf.cast(
        tf.reshape(tf.range(64 * 48 * 100) % 13, [64, 48, 100]), tf.float64
    )
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_batch_matmul_strided_batch_128(self):
    a = tf.cast(
        tf.reshape(tf.range(128 * 4 * 48) % 17, [128, 4, 48]), tf.float64
    )
    b = tf.cast(
        tf.reshape(tf.range(128 * 48 * 100) % 13, [128, 48, 100]), tf.float64
    )
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_batch_matmul_strided_transpose_a_broadcast_b(self):
    a = tf.cast(
        tf.reshape(tf.range(64 * 48 * 4) % 17, [64, 48, 4]), tf.float64
    )
    b = tf.cast(tf.reshape(tf.range(48 * 100) % 13, [48, 100]), tf.float64)
    self._compare(
        lambda x, y: tf.matmul(x, y, transpose_a=True), [a, b]
    )

  def test_batch_matmul_strided_transpose_b(self):
    a = tf.cast(
        tf.reshape(tf.range(64 * 4 * 48) % 17, [64, 4, 48]), tf.float64
    )
    b = tf.cast(
        tf.reshape(tf.range(64 * 100 * 48) % 13, [64, 100, 48]), tf.float64
    )
    self._compare(
        lambda x, y: tf.matmul(x, y, transpose_b=True), [a, b]
    )

  def test_batch_matmul_strided_4d_no_broadcast(self):
    a = tf.cast(
        tf.reshape(tf.range(2 * 3 * 4 * 5) % 7, [2, 3, 4, 5]), tf.float64
    )
    b = tf.cast(
        tf.reshape(tf.range(2 * 3 * 5 * 6) % 11, [2, 3, 5, 6]), tf.float64
    )
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_batch_matmul_nonsimple_broadcast_fallback(self):
    a = tf.cast(
        tf.reshape(tf.range(2 * 1 * 4 * 5) % 7, [2, 1, 4, 5]), tf.float64
    )
    b = tf.cast(
        tf.reshape(tf.range(1 * 3 * 5 * 6) % 11, [1, 3, 5, 6]), tf.float64
    )
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_batch_matmul_zero_batch(self):
    a = tf.zeros([0, 4, 5], dtype=tf.float64)
    b = tf.zeros([0, 5, 6], dtype=tf.float64)
    self._compare(lambda x, y: tf.matmul(x, y), [a, b])

  def test_tanh(self):
    x = tf.constant([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=tf.float64)
    self._compare(tf.tanh, [x])

  def test_tanh_double_extreme_range(self):
    x = tf.constant(
        [-100.0, -22.0, -20.0, -10.0, -1.0, -1e-8, 0.0, 1e-8,
         1.0, 10.0, 20.0, 22.0, 100.0],
        dtype=tf.float64,
    )
    self._compare(tf.tanh, [x], rtol=0.0, atol=5e-16)

  def test_tanh_double_large_vector(self):
    x = tf.cast(tf.linspace(-10.0, 10.0, 1000003), tf.float64)
    self._compare(tf.tanh, [x], rtol=0.0, atol=5e-16)

  def test_tanh_double_special_values(self):
    x = tf.constant(
        [
            float("-inf"),
            -20.001,
            -20.0,
            -19.999,
            -0.0,
            0.0,
            19.999,
            20.0,
            20.001,
            float("inf"),
        ],
        dtype=tf.float64,
    )
    self._compare(tf.tanh, [x], rtol=0.0, atol=5e-16)

  def test_tanh_double_nan(self):
    with tf.device("/device:MUSA:0"):
      y = tf.tanh(tf.constant([float("nan")], dtype=tf.float64))
    self.assertTrue(bool(self.evaluate(tf.math.is_nan(y))[0]))

  def test_tanh_double_signed_zero(self):
    with tf.device("/device:MUSA:0"):
      y = tf.tanh(tf.constant([-0.0, 0.0], dtype=tf.float64))
    values = self.evaluate(y)
    self.assertTrue(bool(np.signbit(values[0])))
    self.assertFalse(bool(np.signbit(values[1])))

  def test_tanh_grad(self):
    y = tf.constant([-0.5, 0.0, 0.5], dtype=tf.float64)
    dy = tf.constant([2.0, 3.0, 4.0], dtype=tf.float64)
    self._compare(
        lambda y_in, dy_in: tf.raw_ops.TanhGrad(y=y_in, dy=dy_in),
        [y, dy],
    )

  def test_mul_broadcast(self):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=tf.float64)
    b = tf.constant([2.0, 3.0, 4.0], dtype=tf.float64)
    self._compare(tf.multiply, [a, b])

  def test_realdiv_broadcast(self):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=tf.float64)
    b = tf.constant([2.0, 3.0, 4.0], dtype=tf.float64)
    self._compare(tf.math.divide, [a, b])

  def test_strided_slice_step_and_reverse(self):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=tf.float64)
    self._compare(lambda x: x[::2, ::-1], [a])

  def test_strided_slice_shrink_axis(self):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=tf.float64)
    self._compare(lambda x: x[:, 0], [a])

  def test_repeated_mixed_stream_ordering(self):
    a = tf.constant(
        [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8],
         [0.9, 1.0, 1.1, 1.2],
         [1.3, 1.4, 1.5, 1.6]],
        dtype=tf.float64,
    )
    w = tf.eye(4, dtype=tf.float64)

    def chain(x, weight):
      y = x
      for _ in range(20):
        y = tf.matmul(y, weight)
        y = tf.tanh(y)
        y = tf.multiply(y, 1.5)
        y = tf.math.divide(y, 1.5)
      return y

    self._compare(chain, [a, w])


if __name__ == "__main__":
  tf.test.main()
