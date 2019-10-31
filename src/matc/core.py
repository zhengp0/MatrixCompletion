# -*- coding: utf-8 -*-
"""
    core
    ~~~~

    core module of matc package.
"""
import numpy as np
from . import utils


class CovariatesNode:
    def __init__(self, covs, sigma):
        """
        This node specifies the matrix predicted based on fixed effect covariates
        `covs`, and weights the resulting matrix by 1/sigma^2.
        
        :param covs: (np.ndarray)
        :param sigma: (int)
        """
        assert isinstance(sigma, int)
        assert covs.ndim == 3
        assert sigma > 0.0
        self.covs = covs
        self.sigma = sigma
        self.num_covs = self.covs.shape[0]
        self.shape = self.covs.shape[1:]

        self.covs_mat = self.covs.reshape(self.num_covs, np.prod(self.shape)).T
        self.weight_mat = np.empty(self.shape)
        self.weight_mat.fill(1.0/sigma**2)
        self.alpha = np.zeros(self.num_covs)

    @property
    def predict_mat(self):
        """
        Predicts the matrix using the covariates
        in self.covs_mat with the updated coefficients
        stored in self.alpha.
        """
        mat = self.covs_mat.dot(self.alpha)
        return mat.reshape(self.shape)

    def update_params(self, mat):
        """
        Estimates the coefficients on the covariates and thereby updates the
        matrix parameters. Estimation done by least squares.

        :param mat: (np.ndarray)
        """
        assert mat.shape == self.shape
        vec = mat.reshape(mat.size)
        self.alpha = np.linalg.solve(self.covs_mat.T.dot(self.covs_mat),
                                     self.covs_mat.T.dot(vec))


class DataNode:
    def __init__(self, mask, data, sigma):
        """
        This node matches the predicted matrix with the observed data,
        but "masks" predicted matrix so that we are only using entries
        that were actually observed from the matrix. Weights the
        resulting matrix by 1/sigma^2.

        :param mask: (utils.Mask)
        :param data: (np.ndarray)
        :param sigma: (int)
        """
        assert isinstance(sigma, int)
        assert sigma > 0.0
        assert isinstance(mask, utils.Mask)
        assert isinstance(data, np.ndarray)
        assert mask.size == data.size
        self.mask = mask
        self.data = data
        self.sigma = sigma
        self.shape = self.mask.shape

        self.weight_mat = self.mask.inv(np.repeat(1.0/sigma**2, self.mask.size))

    @property
    def predict_mat(self):
        return self.mask.inv(self.data)

    def update_params(self, mat):
        assert mat.shape == self.shape
        pass


class LowRankNode:
    def __init__(self, shape, rank, sigma):
        """
        This node specifies a low-rank structure with rank `rank` for the predicted
        matrix. Weights the resulting matrix by 1/sigma^2.
        
        :param shape: (int, int)
        :param rank: (int)
        :param sigma: (int)
        """
        assert isinstance(sigma, int)
        assert sigma > 0.0
        assert isinstance(shape, tuple)
        assert isinstance(rank, int)
        assert rank > 0
        assert rank <= min(*shape)
        self.shape = shape
        self.rank = rank
        self.sigma = sigma

        self.weight_mat = np.empty(self.shape)
        self.weight_mat.fill(1.0/sigma**2)

        self.u = np.zeros((self.shape[0], self.rank))
        self.v = np.zeros((self.shape[1], self.rank))

    @property
    def predict_mat(self):
        """
        Gives the full predicted matrix based on the low rank
        u and v.
        """
        return self.u.dot(self.v.T)

    def update_params(self, mat):
        """
        Updates the low rank structure of self.u
        and self.v based on the `mat` input. Performs
        singular value decomposition on the matrix and selects
        the last `self.rank` entries.

        :param mat: (np.ndarray)
        """
        assert mat.shape == self.shape
        a, s, b = np.linalg.svd(mat, full_matrices=False)
        self.u = a[:, :self.rank]*s[:self.rank]
        self.v = b.T[:, :self.rank]
