# -*- coding: utf-8 -*-
"""
    test_utils
    ~~~~~~~~~~

    Test for the utils module.
"""
import numpy as np
import pytest
from matc import utils


@pytest.mark.parametrize("shape", [(3, 3)])
@pytest.mark.parametrize("index", [(np.arange(3), np.arange(3))])
@pytest.mark.parametrize("mat", [np.eye(3), np.random.randn(3, 3)])
def test_utils_mask(shape, index, mat):
    mask = utils.Mask(shape, index)
    assert np.linalg.norm(mask(mat) - np.diag(mat)) < 1e-10


@pytest.mark.parametrize("shape", [(3, 3)])
@pytest.mark.parametrize("index", [(np.arange(3), np.arange(3))])
@pytest.mark.parametrize("vec", [np.ones(3), np.random.randn(3)])
def test_utils_mask_inv(shape, index, vec):
    mask = utils.Mask(shape, index)
    assert np.linalg.norm(mask.inv(vec) - np.diag(vec)) < 1e-10
