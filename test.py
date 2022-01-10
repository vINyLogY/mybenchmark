# coding: utf-8
import numpy as np
import tensorflow as tf
from scipy import linalg
from functools import wraps, partial
from time import time


def time_this(func):
    @wraps(func)
    def _time_this(*args, **kwargs):
        start = time()
        r = func(*args, **kwargs)
        end = time()
        print('{}.{} : {}', func.__module__, func.__name__, end - start)
        return r

    return _time_this


if __name__ == '__main__':
    DIM = 10_000
    seed = np.random.rand(DIM, DIM)
    np_a = np.array(seed + seed.T, dtype=complex)
    tf_a = tf.constant(np_a)
    np_eigh = np.linalg.eigh
    sp_eigh = linalg.eigh
    tf_eigh = tf.linalg.eigh

    @time_this
    def test_np():
        return np_eigh(np_a)

    @time_this
    def test_sp():
        return sp_eigh(np_a)

    @time_this
    def test_tf():
        return tf_eigh(tf_a)

    test_np()
    test_sp()
    test_tf()
