# -*- coding: utf-8 -*-
"""
    test_core
    ~~~~~~~~~

    Test core module.
"""
import numpy as np
import pytest
from matc import core
from matc import utils


@pytest.mark.parametrize("covs", [np.random.randn(3, 5, 4)])
@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_core_covariates_node(covs, sigma):
    covs_node = core.CovariatesNode(covs, sigma)
    assert covs_node.num_covs == 3
    assert covs_node.shape == (5, 4)
    assert covs_node.covs_mat.shape == (20, 3)
    assert np.all(covs_node.weight_mat == 1.0/sigma**2)
    assert np.all(covs_node.alpha == 0.0)


@pytest.mark.parametrize("covs", [np.random.randn(3, 5, 4)])
@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_core_covariates_node_predict_mat(covs, sigma):
    covs_node = core.CovariatesNode(covs, sigma)
    predict_mat = covs_node.predict_mat
    assert predict_mat.shape == (5, 4)
    assert np.linalg.norm(predict_mat) < 1e-10


@pytest.mark.parametrize("covs", [np.random.randn(3, 5, 4)])
@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_core_covariates_node_update_params(covs, sigma):
    covs_node = core.CovariatesNode(covs, sigma)
    mat = np.random.randn(*covs_node.shape)
    vec = mat.reshape(mat.size)
    covs_node.update_params(mat)
    c = covs_node.covs_mat.T.dot(covs_node.covs_mat)
    y = covs_node.covs_mat.T.dot(vec)
    assert np.linalg.norm(c.dot(covs_node.alpha) - y) < 1e-10


@pytest.mark.parametrize("mask",
                         [utils.Mask((3, 3), (np.arange(3), np.arange(3)))])
@pytest.mark.parametrize("data", [np.random.randn(3)])
@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_core_data_node(mask, data, sigma):
    data_node = core.DataNode(mask, data, sigma)
    assert np.linalg.norm(data_node.weight_mat -
                          np.diag(np.repeat(1.0/sigma**2, 3))) < 1e-10
    assert np.linalg.norm(data_node.predict_mat - np.diag(data)) < 1e-10
