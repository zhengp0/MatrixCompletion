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


@pytest.mark.parametrize("shape", [(5, 4)])
@pytest.mark.parametrize("rank", [2])
@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_core_low_rank_node(shape, rank, sigma):
    low_rank_node = core.LowRankNode(shape, rank, sigma)
    predict_mat = low_rank_node.predict_mat
    assert predict_mat.shape == shape
    assert np.linalg.norm(predict_mat -
                          low_rank_node.u.dot(low_rank_node.v.T)) < 1e-10

    mat = np.random.randn(*shape)
    low_rank_node.update_params(mat)
    a, s, b = np.linalg.svd(mat)
    mat_approx = (a[:, :rank]*s[:rank]).dot(b[:rank])
    assert np.linalg.norm(low_rank_node.predict_mat - mat_approx) < 1e-10


@pytest.fixture
def shape():
    return 3, 3


@pytest.fixture
def data_node(shape):
    index = (np.arange(3), np.arange(3))
    mask = utils.Mask(shape, index)
    data = np.ones(3)
    sigma = 0.1
    return core.DataNode(mask, data, sigma)


@pytest.fixture
def covs_node(shape):
    covs = np.ones((1,) + shape)
    sigma = 0.1
    return core.CovariatesNode(covs, sigma)


def test_core_matrix_completion(data_node, covs_node):
    mc = core.MatrixCompletion([data_node, covs_node])
    covs_node.alpha[:] = 1.0
    objective = 0.5*(np.sum(data_node.predict_mat**2)/data_node.sigma**2 +
                     np.sum(covs_node.predict_mat**2)/covs_node.sigma**2)
    assert np.abs(mc.objective() - objective) < 1e-10
    mc.update_mat()
    assert np.linalg.norm(mc.mat - 1.0) < 1e-10
