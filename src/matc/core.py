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
        self.weight_mat = np.empty(self.shape)
        self.weight_mat.fill(1.0/sigma**2)
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


class DataNode:
    def __init__(self, mask, data, sigma):
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
        return self.u.dot(self.v.T)

    def update_params(self, mat):
        assert mat.shape == self.shape
        a, s, b = np.linalg.svd(mat, full_matrices=False)
        self.u = a[:, :self.rank]*s[:self.rank]
        self.v = b.T[:, :self.rank]


class MatrixCompletion:
    def __init__(self, nodes):
        assert isinstance(nodes, list)
        assert len(nodes) != 0
        self.nodes = nodes
        self.num_nodes = len(self.nodes)
        self.shape = self.nodes[0].shape
        self.mat = np.zeros(self.shape)

    def objective(self):
        val = 0.0
        for node in self.nodes:
            val += 0.5*np.sum(node.weight_mat*(node.predict_mat - self.mat)**2)
        return val

    def update_mat(self):
        weight_mats = np.array([node.weight_mat for node in self.nodes])
        predict_mats = np.array([node.predict_mat for node in self.nodes])

        self.mat = np.sum(weight_mats*predict_mats, axis=0) /\
            np.sum(weight_mats, axis=0)

    def complete_matrix(self,
                        verbose=False,
                        max_iter=100,
                        tol=1e-6):
        iter_count = 0
        err = tol + 1
        mat_old = self.mat.copy()
        while err >= tol:
            # update mat
            self.update_mat()

            # update node params
            for node in self.nodes:
                node.update_params()

            # update iteration information
            err = np.linalg.norm(self.mat - mat_old)/self.mat.size
            iter_count += 1
            if verbose:
                obj = self.objective()
                print("iter %5d, obj %7.2e, err %7.2e" %
                      (iter_count, obj, err))

            if iter_count >= max_iter:
                print("Reach maximum iteration.")
                break
