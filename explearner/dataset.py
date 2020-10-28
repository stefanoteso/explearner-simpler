import os
import requests

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn import tree, preprocessing
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm, kendalltau
from scipy.spatial.distance import cosine as cosine_dist
from itertools import product, combinations
from os.path import join
from sympy.utilities.iterables import multiset_permutations

from treeinterpreter import treeinterpreter as ti

from . import KendallKernel, rank_items, kendall_tau_dist, dump
from .kernel import CombinerKernel


# TODO move from X, Z, y to context, best_arm.


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


_COLORS = [
    (255, 0, 0),  # r
    (0, 255, 0),  # g
    (0, 128, 255),  # b
    (128, 0, 255),  # v
]
_COLOR_TO_INDEX = {tuple(color): i for i, color in enumerate(_COLORS)}
_RULE_TO_COORDS = {
    0: [[0, 0], [0, 4], [4, 0], [4, 4]],
    1: [[0, 1], [0, 2], [0, 3]]
}


class ColorsDataset(Dataset):
    """Toy dataset used by RRR and CAIPI."""

    def __init__(self, **kwargs):
        self.rule = kwargs.pop('rule')
        self.kind = kwargs.pop('kind', 'relevance')

        path = join('data', f'toy_colors_{self.kind}_{self.rule}.pickle')
        try:
            X, Z, y = load(path)
        except:
            data = np.load(join('data', 'toy_colors.npz'))

            X_tr = np.array([self._img_to_x(img) for img in data['arr_0']])
            X_ts = np.array([self._img_to_x(img) for img in data['arr_1']])
            X = np.vstack([X_tr, X_ts])

            y_tr = np.array([self._classify(x, self.rule) for x in X_tr])
            y_ts = np.array([self._classify(x, self.rule) for x in X_ts])
            y = np.hstack([y_tr, y_ts])

            if self.kind == 'relevance':
                Z = np.array([self._explain_relevance(x, self.rule) for x in X])
            elif self.kind == 'polarity':
                Z = np.array([self._explain_polarity(x, self.rule) for x in X])
            else:
                raise ValueError(f'kind must be "relevance" or "polarity"')

            dump(path, (X, Z, y))

        # XXX
        X = X[:10]
        Z = Z[:10]
        y = y[:10]

        # FIXME: for x and z use k-pixel kernels
        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = RBF(length_scale=1, length_scale_bounds=(1, 1))
        ky = RBF(length_scale=1, length_scale_bounds=(1, 1))

        if self.kind == 'relevance':
            arms = self._enumerate_relevance_arms(self.rule)
        else:
            arms = self._enumerate_polairty_arms(self.rule)

        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)

        # XXX keep only explanations with "extreme" rewards
        self.arms = [arm for arm in self.arms
                     if np.abs(self.reward(0, arm[0], arm[1])) >= 0.5]

    @staticmethod
    def _img_to_x(img):
        img = img.reshape((5, 5, 3))
        x = [_COLOR_TO_INDEX[tuple(img[r, c])]
             for r, c in product(range(5), repeat=2)]
        return np.array(x, dtype=np.float32)

    @staticmethod
    def _classify(x, rule):
        """Computes the ground-truth label."""
        x = x.reshape((5, 5))
        if rule == 0:
            r = int(x[0, 0] == x[0, 4] and
                    x[0, 0] == x[4, 0] and
                    x[0, 0] == x[4, 4])
        else:
            r = int(x[0, 1] != x[0, 2] and
                    x[0, 1] != x[0, 3] and
                    x[0, 2] != x[0, 3])
        return 2 * r - 1

    @staticmethod
    def _explain_relevance(x, rule):
        """Computes the relevance-based ground-truth explanation.  Notice
        that relevance explanations are independent of x and y."""
        x = x.reshape((5, 5))
        coords = _RULE_TO_COORDS[rule]

        z = np.zeros_like(x, dtype=np.int8)
        for r, c in coords:
            z[r, c] = 1
        return z.ravel()

    @staticmethod
    def _explain_polarity(x, rule):
        """Computes the polarity-based ground-truth explanation."""
        x = x.reshape((5, 5))
        coords = _RULE_TO_COORDS[rule]

        counts = np.bincount([x[r, c] for r, c in coords])
        max_count, max_value = np.max(counts), np.argmax(counts)

        z = np.zeros_like(x, dtype=np.int8)
        if rule == 0:
            for r, c in coords:
                weight = 1 if max_count != 1 and x[r, c] == max_value else -1
                z[r, c] = weight
        else:
            for r, c in coords:
                weight = 1 if max_count == 1 or x[r, c] != max_value else -1
                z[r, c] = weight
        return z.ravel()

    @staticmethod
    def _enumerate_relevance_arms(rule):
        """Enumerate all masks of k pixels."""
        k = 4 if rule == 0 else 3

        coords = product(range(5), repeat=2)
        ktuples = combinations(coords, k)

        arms = []
        for ktuple, label in product(ktuples, [-1, 1]):
            z = np.zeros((5, 5), dtype=np.int8)
            for r, c in ktuple:
                z[r, c] = 1
            arms.append((z.reshape(5 * 5), label))

        return arms

    @staticmethod
    def _enumerate_polarity_arms(rule):
        """Enumerate all masks of k pixels with per-pixel pos/neg labels."""
        k = 4 if rule == 0 else 3

        coords = product(range(5), repeat=2)
        ktuples = combinations(coords, k)
        signs = product(*(([-1, 1],) * k))

        arms = []
        for ktuple, ktuple_sign, label in product(ktuples, signs, [-1, 1]):
            z = np.zeros((5, 5), dtype=np.int8)
            for (r, c), s in zip(ktuple, ktuple_sign):
                z[r, c] = s
            arms.append((z.reshape(5 * 5), label))

        return arms

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i], self.y[i]
        sign = 1 if y == yhat else -1
        return sign * (1 - cosine_dist(z, zhat)) + self.rng.normal(0, noise)


class AdultDataset(Dataset):
    def __init__(self, **kwargs):
        from shap.datasets import adult
        from sklearn.svm import LinearSVC

        clf_type = kwargs.pop('clf')
        X, y = adult(display=False)
        n_examples, n_features = X.shape

        if clf_type == 'dt':
            raise NotImplementedError()
        elif clf_type == 'lm':
            grid = {'C': np.logspace(-2, 2, 5)}
            clf = LinearSVC(penalty='l1', dual=False, random_state=0)
            clf = self.select_model(clf, X.values, y, grid)
            global_z = (clf.coef_ != 0).astype(float)[0]
            Z = np.array([global_z] * n_examples).reshape((n_examples, -1))
        else:
            raise ValueError()

        raise NotImplementedError()

    def reward(self, i, zhat, yhat, noise=0):
        raise NotImplementedError()


class TreeDataset(Dataset):
    @abstractmethod
    def reward(self, i, zhat, yhat, noise=0):
        pass

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

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i], self.y[i]

        sign = 1 if y == yhat else -1
        return sign * (jaccard_score(z, zhat)) + self.rng.normal(0, noise)


class BanknoteAuth(TreeDataset):
    def __init__(self, **kwargs):
        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"]
        self.load_dataset('data', urls)

        # total 1372 instances, 610 1s = fake, 762 0s = real
        columns = ["variance", "skew", "curtosis", "entropy", "class"]
        dataset = pd.read_csv("data/data_banknote_authentication.txt", names=columns)

        # target values: 1 is fake, 0 is real
        y = dataset['class'].to_numpy()
        # creating the feature vector
        X = dataset.drop('class', axis=1).to_numpy()

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y)
        self.tree = clf.tree_
        # Extremely sparse explanations
        Z = clf.decision_path(X).toarray()

        # Avoid sparse features - Maybe in the future
        # decision_path = clf.decision_path(X).toarray()
        # for i in np.arange(len(decision_path.indptr) - 1):
        #     decision_path.indices[decision_path.indptr[i]:decision_path.indptr[i+1]]
        # Z = np.array(Z)

        # Kernels
        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = DotProduct(sigma_0=1, sigma_0_bounds=(1, 1))  # Explanations are sparse
        ky = DotProduct(sigma_0=1, sigma_0_bounds=(1, 1))

        # Extract all possible root-to-leaf explanations
        arms_z = self.root_to_leaf_paths(0)
        arms_y = np.array([0, 1])
        arms = list(product(arms_z, arms_y))
        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)

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


class BreastCancer(Dataset):
    """
    Breast cancer dataset with ranking explanations extracted from a Random Forest classifier.
    """

    def __init__(self, **kwargs):
        # Samples per class	212(M),357(B)
        dataset = load_breast_cancer()
        pca_dim = 10

        # Data
        X = dataset.data
        # 0 is "malignant", 1 is "benign"
        y = dataset.target

        # We reduce the dimensionality to be able to generate all possible rankings as explanations
        normalized_data = StandardScaler().fit_transform(X)
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(normalized_data)

        rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        rf.fit(X, y)
        _, _, contributions = ti.predict(rf, X)

        Z = np.array([rank_items(contr) if y[i] else rank_items(-contr) for i, contr in enumerate(contributions[:, :, 0])])

        # Kernels
        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = KendallKernel()  # ranking kernels
        ky = DotProduct(sigma_0=1, sigma_0_bounds=(1, 1))

        # The space of explanations is all possible permutations!
        # TODO: Improve efficiency by excluding some permutation
        arms_z = np.array(list(multiset_permutations(np.arange(pca_dim))))
        arms_y = np.array([0, 1])
        arms = list(product(arms_z, arms_y))

        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i], self.y[i]
        sign = 1 if y == yhat else -1
        return sign * (kendall_tau_dist(z, zhat)) + self.rng.normal(0, noise)


class WineQuality(TreeDataset):
    """
    Wine quality is a larger regression dataset to test regression reward.
    Samples: 4898, Features: 12
    """

    def __init__(self, **kwargs):
        dataset = pd.read_csv("data/winequality-white.csv", sep=';')

        # target values: wine quality, values from 3-9
        y = dataset['quality'].to_numpy()
        # creating the feature vector
        X = dataset.drop('quality', axis=1).to_numpy()
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)

        # clf_type = kwargs.pop('clf')

        clf = tree.DecisionTreeRegressor()
        clf = clf.fit(X, y)
        self.tree = clf.tree_
        # Extremely sparse explanations
        Z = clf.decision_path(X).toarray()

        # Kernels
        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = DotProduct(sigma_0=1, sigma_0_bounds=(1, 1))
        ky = RBF(length_scale=1, length_scale_bounds=(1, 1))

        arms_z = self.root_to_leaf_paths(0)
        arms_y = np.arange(3, 10)
        arms = list(product(arms_z, arms_y))
        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)

    def reward(self, i, zhat, yhat, noise=0):
        z, y = self.Z[i], self.y[i]
        reward_y = norm(loc=y, scale=0.1).pdf(yhat)
        return reward_y * jaccard_score(z, zhat) + self.rng.normal(0, noise)
