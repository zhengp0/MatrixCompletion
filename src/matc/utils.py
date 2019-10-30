# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~

    utils module for matc package, contains utility functions and class.
"""
import numpy as np


class Mask:
    def __init__(self, shape, index):
        assert isinstance(shape, tuple)
        assert isinstance(index, tuple)

        self.shape = shape
        self.index = index
        self.ndim = len(self.index)
        self.size = len(self.index[0])

    def __call__(self, mat):
        assert mat.shape == self.shape
        return mat[self.index]

    def inv(self, vec):
        assert vec.size == self.size
        mat = np.zeros(self.shape)
        mat[self.index] = vec
        return mat
