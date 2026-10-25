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

"""MUSA ConcatV2 double coverage tests."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ConcatV2DoubleTest(MUSATestCase):
  def _check(self, tensors, axis, axis_dtype=tf.int32):
    axis_tensor = tf.constant(axis, dtype=axis_dtype)

    def op(*values):
      return tf.raw_ops.ConcatV2(values=list(values), axis=axis_tensor)

    self._compare_cpu_musa_results(
        op, tensors, tf.float64, rtol=0.0, atol=0.0
    )

  def test_axis_0_three_inputs(self):
    self._check(
        [
            tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64),
            tf.constant([[5.0, 6.0]], dtype=tf.float64),
            tf.constant([[7.0, 8.0], [9.0, 10.0]], dtype=tf.float64),
        ],
        axis=0,
    )

  def test_axis_1_two_inputs(self):
    self._check(
        [
            tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64),
            tf.constant([[5.0], [6.0]], dtype=tf.float64),
        ],
        axis=1,
    )

  def test_negative_axis(self):
    self._check(
        [
            tf.constant([[[1.0], [2.0]]], dtype=tf.float64),
            tf.constant([[[3.0], [4.0]]], dtype=tf.float64),
        ],
        axis=-1,
    )

  def test_int64_axis(self):
    self._check(
        [
            tf.constant([[1.0, 2.0]], dtype=tf.float64),
            tf.constant([[3.0, 4.0]], dtype=tf.float64),
        ],
        axis=0,
        axis_dtype=tf.int64,
    )

  def test_axis_1_outer_loop_downstream_add(self):
    """Exercise multiple D2D copies followed by a dependent MUSA op."""
    left = tf.constant(
        np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)
    )
    right = tf.constant(
        np.arange(4 * 2 * 5, dtype=np.float64).reshape(4, 2, 5) + 1000.0
    )
    axis_tensor = tf.constant(1, dtype=tf.int32)

    def op(*values):
      concatenated = tf.raw_ops.ConcatV2(
          values=list(values), axis=axis_tensor
      )
      return concatenated + 0.5

    self._compare_cpu_musa_results(
        op, [left, right], tf.float64, rtol=0.0, atol=0.0
    )

  def test_zero_element_axis0(self):
    self._check(
        [
            tf.zeros([0, 2], dtype=tf.float64),
            tf.zeros([0, 2], dtype=tf.float64),
        ],
        axis=0,
    )

  def test_zero_element_non_axis_dim(self):
    self._check(
        [
            tf.zeros([2, 0, 3], dtype=tf.float64),
            tf.zeros([2, 0, 4], dtype=tf.float64),
        ],
        axis=2,
    )

  def test_single_non_empty_input(self):
    self._check(
        [
            tf.zeros([0, 2], dtype=tf.float64),
            tf.ones([1, 2], dtype=tf.float64),
        ],
        axis=0,
    )

  def test_four_dimensional_axes(self):
    left = tf.reshape(tf.range(2 * 3 * 4 * 5, dtype=tf.float64),
                      [2, 3, 4, 5])
    right = tf.reshape(tf.range(2 * 2 * 4 * 5, dtype=tf.float64) + 1000.0,
                       [2, 2, 4, 5])
    self._check([left, right], axis=1)
    self._check([left, right], axis=-3)


if __name__ == "__main__":
  tf.test.main()
