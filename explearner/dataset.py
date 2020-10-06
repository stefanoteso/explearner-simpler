import numpy as np

from abc import ABC, abstractmethod
from sklearn.utils import check_random_state
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from scipy.spatial.distance import cosine as cosine_dist
from itertools import product, combinations
from os.path import join


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
    ks : tuple
        The kernels over contexts, explanations, and labels.
    arms : list of (ndarray of shape (n_exp_variables,), scalar) pairs
        The possible (explanation, label) pairs.
    combiner : str or callable, defaults to 'prod'
        How to combine the three kernels.
    rng : int or RandomState or None
        The RNG.

    Attributes
    ----------
    f : ndarray of shape (n_examples,)
        Pre-computed reward for the optimal arm (noiseless).
    """
    def __init__(self, X, Z, y, ks, arms, combiner='prod', rng=None):
        assert X.ndim == 2 and Z.ndim == 2 and y.ndim == 1
        assert arms[0][0].ndim == 1
        self.rng = check_random_state(rng)

        self.X, self.Z, self.y = X, Z, y
        self.kx, self.kz, self.ky = ks
        self.arms = arms

        if combiner == 'sum':
            combiner = lambda kx, kz, ky: (kx + kz) * ky
        elif combiner == 'prod':
            combiner = lambda kx, kz, ky: kx * kz * ky
        else:
            raise ValueError(f'unknown combiner {combiner}')
        self.kernel = combiner(self.kx, self.kz, self.ky)

        self.f = np.array([self.reward(i, Z[i], y[i], noise=0)
                           for i in range(self.X.shape[0])])

    @abstractmethod
    def reward(self, i, zhat, yhat, noise=0):
        """Reward of (x[i], zhat, yhat) given that the true triple is
        (x[i], z[i], y[i])."""
        pass

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


class SineDataset(Dataset):
    """Toy 1-D dataset."""
    def __init__(self, min_x=0, max_x=30, n=1000, **kwargs):
        X = np.random.uniform(min_x, max_x, size=n).reshape(-1, 1)
        Z = -np.cos(X * 0.5) # XXX really?
        y = np.sin(X * 0.5).ravel()

        kx = gpflow.kernels.RBF(active_dims=[0])
        kz = gpflow.kernels.Cosine(active_dims=[1])
        ky = gpflow.kernels.Linear(active_dims=[2])

        arms_z = list(np.arange(-2, 2, 0.05).reshape(-1, 1))
        arms_y = list(np.arange(-2, 2, 0.05))
        arms = list(product(arms_z, arms_y))
        super().__init__(X, Z, y, (kx, kz, ky), arms, **kwargs)

    def reward(self, i, zhat, yhat, noise=0):
        x, z, y = self.X[i,0], self.Z[i,0], self.y[i]
        zhat = zhat[0]
        reward_z = norm(loc=z, scale=1).pdf(zhat)
        reward_y = norm(loc=y, scale=1).pdf(yhat)
        return reward_z * reward_y + self.rng.normal(0, noise)


_COLORS = [
    (255,   0,   0), # r
    (0,   255,   0), # g
    (0,   128, 255), # b
    (128,   0, 255), # v
]
_COLOR_TO_INDEX = {tuple(color): i for i, color in enumerate(_COLORS)}


class ColorsDataset(Dataset):
    """Toy dataset used by RRR and CAIPI."""
    def __init__(self, **kwargs):
        self.rule = kwargs.pop('rule')

        path = join('data', f'toy_colors_{self.rule}.pickle')
        try:
            X, Z, y = load(path)
        except:
            data = np.load(join('data', 'toy_colors.npz'))

            X_tr = np.array([self._img_to_x(img) for img in data['arr_0']])
            X_ts = np.array([self._img_to_x(img) for img in data['arr_1']])
            X = np.vstack([X_tr, X_ts])

            y_tr = np.array([self._classify(x) for x in X_tr])
            y_ts = np.array([self._classify(x) for x in X_ts])
            y = np.hstack([y_tr, y_ts])

            Z = np.array([self._explain(x) for x in X])
            dump(path, (X, Z, y))

        kx = gpflow.kernels.RBF(active_dims=list(range(0, 25)))
        kz = gpflow.kernels.RBF(active_dims=list(range(25, 50)))
        ky = gpflow.kernels.Linear(active_dims=[50])

        arms = self._enumerate_arms()
        super().__init__(X, Z, y, (kx, kz, ky), arms, **kwargs)

    @staticmethod
    def _img_to_x(img):
        img = img.reshape((5, 5, 3))
        x = [_COLOR_TO_INDEX[tuple(img[r,c])]
             for r, c in product(range(5), repeat=2)]
        return np.array(x, dtype=np.float32)

    @staticmethod
    def _rule0(x):
        return int(x[0,0] == x[0,4] and x[0,0] == x[4,0] and x[0,0] == x[4,4])

    @staticmethod
    def _rule1(x):
        return int(x[0,1] != x[0,2] and x[0,1] != x[0,3] and x[0,2] != x[0,3])

    def _classify(self, x):
        return {0: self._rule0, 1: self._rule1}[self.rule](x.reshape((5, 5)))

    def _explain(self, x):
        coords = {
            0: [[0, 0], [0, 4], [4, 0], [4, 4]],
            1: [[0, 1], [0, 2], [0, 3]]
        }[self.rule]

        x = x.reshape((5, 5))
        counts = np.bincount([x[r,c] for r, c in coords])
        max_count, max_value = np.max(counts), np.argmax(counts)

        z = np.zeros_like(x, dtype=np.float32)
        if self.rule == 0:
            for r, c in coords:
                weight = 1 if max_count != 1 and x[r, c] == max_value else -1
                z[r, c] = weight
        else:
            for r, c in coords:
                weight = 1 if max_count == 1 or x[r, c] != max_value else -1
                z[r, c] = weight
        return z.ravel()

    def _enumerate_arms(self):
        """Enumerate all possible (explanation, label) pairs.  Basically it
        enumerates all possible 3 or 4 pixels depending on the rule and assigns
        them all possible signs."""
        n_coords = 4 if self.rule == 0 else 3
        polarities = list(product(*(([0, 1],) * n_coords)))

        arms = []
        all_coords = product(range(5), repeat=2)
        for coords in combinations(all_coords, n_coords):
            for signs in polarities:
                for label in [-1, 1]:
                    z = np.zeros((5, 5), dtype=np.int8)
                    for (r, c), s in zip(coords, signs):
                        z[r, c] = 2 * s - 1
                    arms.append((z.reshape(5*5), label))

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
