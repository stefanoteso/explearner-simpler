import numpy as np

from os.path import join
from sklearn.gaussian_process.kernels import RBF
from itertools import product, combinations

from . import Dataset, EqCosRewardMixin
from .. import load, dump


_COLOR_TO_OHE = {
    (255, 0, 0):   (1, 0, 0, 0),  # r
    (0, 255, 0):   (0, 1, 0, 0),  # g
    (0, 128, 255): (0, 0, 1, 0),  # b
    (128, 0, 255): (0, 0, 0, 1),  # v
}
_OHE_TO_VALUE = {
    (1, 0, 0, 0): 0,
    (0, 1, 0, 0): 1,
    (0, 0, 1, 0): 2,
    (0, 0, 0, 1): 3,
}
_RULE_TO_COORDS = {
    0: [[0, 0], [0, 4], [4, 0], [4, 4]],
    1: [[0, 1], [0, 2], [0, 3]]
}


class ColorsDataset(EqCosRewardMixin, Dataset):
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

        # TODO: run on all training images
        X = X[:10]
        Z = Z[:10]
        y = y[:10]

        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = RBF(length_scale=1, length_scale_bounds=(1, 1))
        ky = RBF(length_scale=1, length_scale_bounds=(1, 1))

        if self.kind == 'relevance':
            arms = self._enumerate_relevance_arms(self.rule)
        else:
            arms = self._enumerate_polarity_arms(self.rule)

        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)

    @staticmethod
    def _img_to_x(img):
        img = img.reshape((5, 5, 3))
        x = [_COLOR_TO_OHE[tuple(img[r, c])]
             for r, c in product(range(5), repeat=2)]
        return np.array(x, dtype=np.int8).ravel()

    @staticmethod
    def _classify(x, rule):
        """Computes the ground-truth label."""
        x = x.reshape((5, 5, 4))
        if rule == 0:
            r = int((x[0, 0] == x[0, 4]).all() and
                    (x[0, 0] == x[4, 0]).all() and
                    (x[0, 0] == x[4, 4]).all())
        else:
            r = int((x[0, 1] != x[0, 2]).all() and
                    (x[0, 1] != x[0, 3]).all() and
                    (x[0, 2] != x[0, 3]).all())
        return 2 * r - 1

    @staticmethod
    def _explain_relevance(x, rule):
        """Computes the relevance-based ground-truth explanation.  Notice
        that relevance explanations are independent of x and y."""
        coords = _RULE_TO_COORDS[rule]

        z = np.zeros((5, 5), dtype=np.int8)
        for r, c in coords:
            z[r, c] = 1
        return z.ravel()

    @staticmethod
    def _explain_polarity(x, rule):
        """Computes the polarity-based ground-truth explanation."""
        coords = _RULE_TO_COORDS[rule]

        x = x.reshape((5, 5, 4))
        pix = np.array([_OHE_TO_VALUE[tuple(x[r, c, :])]
                       for r, c in product(range(5), repeat=2)]).reshape(5, 5)

        counts = np.bincount([pix[r, c] for r, c in coords])
        max_count, max_value = np.max(counts), np.argmax(counts)

        z = np.zeros((5, 5), dtype=np.int8)
        if rule == 0:
            for r, c in coords:
                weight = 1 if max_count != 1 and pix[r, c] == max_value else -1
                z[r, c] = weight
        else:
            for r, c in coords:
                weight = 1 if max_count == 1 or pix[r, c] != max_value else -1
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

