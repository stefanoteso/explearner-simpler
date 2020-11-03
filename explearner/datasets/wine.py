import numpy as np
import pandas as pd

from os.path import join
from itertools import product
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.preprocessing import StandardScaler

from . import TreeDataset, NormJaccardRewardMixin


_URLS = [
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
]


class WineQuality(NormJaccardRewardMixin, TreeDataset):
    '''
    Wine quality is a larger regression dataset to test regression reward.
    Samples: 4898, Features: 12
    '''

    def __init__(self, **kwargs):
        self.load_dataset('data', _URLS)

        dataset = pd.read_csv(join('data', 'winequality-white.csv'), sep=';')

        # target values: wine quality, values from 3-9
        X = dataset.drop('quality', axis=1).to_numpy()
        X = StandardScaler().fit_transform(X)
        y = dataset['quality'].to_numpy()

        grid = {
            'max_depth': [5, 10, None],
            'max_features': ['auto', None],
            'random_state': [0],
        }
        clf = self.select_model(DecisionTreeRegressor(), X, y, grid,
                                scoring='neg_mean_squared_error')
        clf.fit(X, y)
        Z = clf.decision_path(X).toarray().astype(np.int)

        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = RBF(length_scale=1, length_scale_bounds=(1, 1))
        ky = RBF(length_scale=1, length_scale_bounds=(1, 1))

        arms_z = self.root_to_leaf_paths(clf.tree_, 0)
        arms_y = np.arange(3, 10)
        arms = list(product(arms_z, arms_y))

        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)
