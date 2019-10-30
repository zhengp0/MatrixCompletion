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
        assert covs.ndim == 3
        assert sigma > 0.0
        self.covs = covs
        self.sigma = sigma
        self.num_covs = self.covs.shape[0]
        self.shape = self.covs.shape[1:]

        self.covs_mat = self.covs.reshape(self.num_covs, np.prod(self.shape)).T
        self.weight_mat = 1.0/sigma**2*np.ones(self.shape)
        self.alpha = np.zeros(self.num_covs)

    @property
    def predict_mat(self):
        mat = self.covs_mat.dot(self.alpha)
        return mat.reshape(self.shape)

    def update_params(self, mat):
        assert mat.shape == self.shape
        vec = mat.reshape(mat.size)
        self.alpha = np.linalg.solve(self.covs_mat.T.dot(self.covs_mat),
                                     self.covs_mat.T.dot(vec))
