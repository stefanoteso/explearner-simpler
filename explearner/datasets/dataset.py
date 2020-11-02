import os, requests
import numpy as np

from abc import ABC, abstractmethod
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import jaccard_score
from scipy.stats import norm
from sklearn.utils import check_random_state

from .. import CombinerKernel


class Dataset(ABC):
    """A dataset.

    Arguments
    ---------
    X : ndarray of shape (n_examples, n_ctx_variables)
        The contexts.
    Z : ndarray of shape (n_examples, n_exp_variables)
        The explanations.
    y : ndarray of shape (n_examples,)
        The labels.
    kx : sklearn.gaussian_process.Kernel
        The kernel over contexts.
    kz : sklearn.gaussian_process.Kernel
        The kernel over explanations.
    ky : sklearn.gaussian_process.Kernel
        The kernel over labels.
    arms : list of (ndarray of shape (n_exp_variables,), scalar) pairs
        The possible (explanation, label) pairs.
    combiner : str or callable, defaults to 'prod'
        How to combine the three kernels.
    rng : int or RandomState or None
        The RNG.
    """

    def __init__(self, X, Z, y, kx, kz, ky, arms, combiner='prod', rng=None):
        assert X.ndim == 2
        assert Z.ndim == 2
        assert y.ndim == 1
        assert arms[0][0].ndim == 1

        self.rng = check_random_state(rng)

        self.X, self.Z, self.y = X, Z, y
        self.kernel = CombinerKernel(kx, kz, ky,
                                     X.shape[1],
                                     Z.shape[1],
                                     combiner)
        self.arms = arms

    @abstractmethod
    def reward(self, i, zhat, yhat, noise=0):
        """Reward of (x[i], zhat, yhat) given that the best arm is
        (x[i], z[i], y[i])."""
        pass

    def regret(self, i, zhat, yhat):
        """Regret of (x[i], zhat, yhat) given that the best arm is
        (x[i], z[i], y[i])."""
        r = self.reward(i, self.Z[i], self.y[i]) - self.reward(i, zhat, yhat)
        assert r >= 0
        return r

    @staticmethod
    def load_dataset(path, urls):
        """
        Download the content for a list of URLs and save them to a folder
        :param path: The path to the location where the data will be saved
        :param urls: The list of URLs from which the content will be downloaded and saved
        """
        if not os.path.exists(path):
            os.mkdir(path)

        for url in urls:
            data = requests.get(url).content
            filename = os.path.join(path, os.path.basename(url))
            with open(filename, "wb") as file:
                file.write(data)

    def split(self, n_splits):
        """Iterate over folds."""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.rng)
        for tr, ts in kfold.split(self.X):
            ts = self.rng.permutation(ts)[:20] # XXX
            yield tr, ts

    def select_model(self, clf, X, y, grid):
        """Selects a model using grid search."""
        # TODO move to model-based dataset subclass
        clf = GridSearchCV(clf, grid, scoring='f1_micro').fit(X, y)

        print(f'best params: {clf.best_params_}')
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print(f'{mean:0.3f} ± {2 * std:0.3f} for {params}')

        return clf.best_estimator_


class NormNormRewardMixin:
    """Implements a simple reward function for scalar explanations."""

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i, 0], self.y[i]
        reward_z = norm(loc=z, scale=0.1).pdf(zhat[0])
        reward_y = norm(loc=y, scale=0.1).pdf(yhat)
        return reward_z * reward_y + self.rng.normal(0, noise)


class EqCosRewardMixin:
    """Implements a simple reward function for vector explanations."""

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i], self.y[i]
        sign = 1 if y == yhat else -1
        return sign * (1 - cosine_dist(z, zhat)) + self.rng.normal(0, noise)


class EqJaccardRewardMixin:
    """Implements a reward function for set explanations."""

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i], self.y[i]
        sign = 1 if y == yhat else -1
        return sign * jaccard_score(z, zhat) + self.rng.normal(0, noise)


class NormJaccardRewardMixin:
    """Implements a reward function for set explanations."""

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i], self.y[i]
        reward_y = norm(loc=y, scale=0.1).pdf(yhat)
        return reward_y * jaccard_score(z, zhat) + self.rng.normal(0, noise)


class EqKendallRewardMixin:
    """Implements a reward function for rankings."""

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i], self.y[i]
        sign = 1 if y == yhat else -1
        return sign * (kendall_tau_dist(z, zhat)) + self.rng.normal(0, noise)


class TreeDataset(Dataset):
    def root_to_leaf_paths(self, node_id):
        """
        Finds all root-to-leaf paths in a decision tree.
        Returns: All paths from root-to-leaf
        """
        # The node is a leaf
        if self.tree.children_left[node_id] == self.tree.children_right[node_id]:
            path = np.zeros(self.tree.node_count)
            path[node_id] = 1
            return [path]
        # Recursively scan left and right children
        else:
            paths = []
            left_paths = self.root_to_leaf_paths(self.tree.children_left[node_id])
            for path in left_paths:
                path[node_id] = 1
                paths.append(path)
            right_paths = self.root_to_leaf_paths(self.tree.children_right[node_id])
            for path in right_paths:
                path[node_id] = 1
                paths.append(path)
            return paths
