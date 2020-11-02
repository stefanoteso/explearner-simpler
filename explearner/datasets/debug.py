import numpy as np
from itertools import product
from sklearn.gaussian_process.kernels import RBF

from . import Dataset, NormNormRewardMixin


class DebugDataset(NormNormRewardMixin, Dataset):
    """Debug dataset with a single context."""

    def __init__(self, **kwargs):
        X = np.linspace(0, 0, num=1).reshape(-1, 1)
        Z = np.zeros(X.shape[0]).reshape(-1, 1)
        y = np.zeros(X.shape[0])

        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = RBF(length_scale=1, length_scale_bounds=(1, 1))
        ky = RBF(length_scale=1, length_scale_bounds=(1, 1))

        arms_z = np.linspace(-1, 1, 5).reshape(-1, 1)
        arms_y = np.linspace(-1, 1, 5)
        arms = list(product(arms_z, arms_y))

        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)

    def split(self, n_splits):
        the_only_ctx = np.array([0])
        yield the_only_ctx, the_only_ctx


class LineDataset(NormNormRewardMixin, Dataset):
    """Toy 1-D dataset."""

    def __init__(self, **kwargs):
        X = np.linspace(-1, 1, num=5).reshape(-1, 1)
        Z = np.ones(X.shape[0]).reshape(-1, 1)
        y = np.array([np.dot(x, z) for (x, z) in zip(X, Z)])

        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = RBF(length_scale=1, length_scale_bounds=(1, 1))
        ky = RBF(length_scale=1, length_scale_bounds=(1, 1))

        arms_z = np.linspace(-1, 1, 5).reshape(-1, 1)
        arms_y = np.linspace(-1, 1, 5)
        arms = list(product(arms_z, arms_y))

        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)


class SineDataset(NormNormRewardMixin, Dataset):
    """Toy 1-D dataset."""

    def __init__(self, **kwargs):
        X = np.linspace(-3, 3, num=5).reshape(-1, 1)
        Z = -np.cos(X * 0.5)  # XXX really?
        y = np.sin(X * 0.5).ravel()

        # TODO cosine kernel for z
        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = RBF(length_scale=1, length_scale_bounds=(1, 1))
        ky = RBF(length_scale=1, length_scale_bounds=(1, 1))

        arms_z = np.linspace(-1, 1, 5).reshape(-1, 1)
        arms_y = np.linspace(-1, 1, 5)
        arms = list(product(arms_z, arms_y))

        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)
