import argparse
import os
import statistics
import time

import numpy as np
import tensorflow as tf
import tensorflow_musa


def expected_bias_add(x, bias, data_format):
    if data_format == "NCHW":
        return x + bias.reshape((1, -1, 1, 1))
    return x + bias


def run_correctness():
    cases = [
        ((192, 100), "NHWC"),
        ((3, 5, 7), "NHWC"),
        ((4, 8, 16, 32), "NHWC"),
        ((4, 8, 16, 32), "NCHW"),
    ]
    with tf.Graph().as_default():
        placeholders = []
        outputs = []
        for shape, data_format in cases:
            bias_shape = (shape[1],) if data_format == "NCHW" else (shape[-1],)
            x = tf.compat.v1.placeholder(tf.float64, shape=shape)
            bias = tf.compat.v1.placeholder(tf.float64, shape=bias_shape)
            placeholders.append((x, bias))
            outputs.append(
                tf.nn.bias_add(x, bias, data_format=data_format)
            )
        with tf.compat.v1.Session() as sess:
            for index, ((shape, data_format), (x, bias)) in enumerate(
                zip(cases, placeholders)
            ):
                rng = np.random.default_rng(20261028 + index)
                host_x = rng.standard_normal(shape, dtype=np.float64)
                bias_shape = (
                    (shape[1],)
                    if data_format == "NCHW"
                    else (shape[-1],)
                )
                host_bias = rng.standard_normal(
                    bias_shape, dtype=np.float64
                )
                actual = sess.run(
                    outputs[index],
                    feed_dict={x: host_x, bias: host_bias},
                )
                expected = expected_bias_add(
                    host_x, host_bias, data_format
                )
                max_abs = float(np.max(np.abs(actual - expected)))
                print(
                    f"CORRECT shape={shape} format={data_format} "
                    f"max_abs={max_abs:.3e}"
                )
                if max_abs != 0.0:
                    raise RuntimeError("double BiasAdd mismatch")
    print("BIASADD_DOUBLE_CORRECTNESS_PASS")


def run_timing(shape, data_format, iterations):
    rng = np.random.default_rng(20261028)
    host_x = rng.standard_normal(shape, dtype=np.float64)
    bias_shape = (shape[1],) if data_format == "NCHW" else (shape[-1],)
    host_bias = rng.standard_normal(bias_shape, dtype=np.float64)

    graph = tf.Graph()
    with graph.as_default():
        x = tf.compat.v1.placeholder(tf.float64, shape=shape)
        bias = tf.compat.v1.placeholder(tf.float64, shape=bias_shape)
        output = tf.nn.bias_add(x, bias, data_format=data_format)
        with tf.compat.v1.Session(graph=graph) as sess:
            feed = {x: host_x, bias: host_bias}
            for _ in range(20):
                sess.run(output, feed_dict=feed)
            samples = []
            for _ in range(iterations):
                start = time.perf_counter()
                sess.run(output, feed_dict=feed)
                samples.append((time.perf_counter() - start) * 1e6)
    print(
        f"TIMING shape={shape} format={data_format} "
        f"median_us={statistics.median(samples):.3f} "
        f"mean_us={statistics.mean(samples):.3f} "
        f"min_us={min(samples):.3f} max_us={max(samples):.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.correctness:
        run_correctness()
    if args.timing:
        run_timing((192, 100), "NHWC", args.iterations)
        run_timing((512, 100), "NHWC", args.iterations)
    if not args.correctness and not args.timing:
        parser.error("choose --correctness and/or --timing")


if __name__ == "__main__":
    main()
