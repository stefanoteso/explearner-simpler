import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor as GPR


class CGPUCB(GPR):
    def __init__(self, **kwargs):
        self.strategy = kwargs.pop('strategy', 'random')
        self.batch_size = kwargs.pop('batch_size', 1024)
        super().__init__(**kwargs)

    def fit(self, X, Z, y, f):
        """Wrapper around GaussianProcessRegressor.fit."""
        D = np.concatenate((X, Z, y.reshape(-1, 1)), axis=1)
        return super().fit(D, f)

    def predict(self, X, Z, y, **kwargs):
        """Wrapper around GaussianProcessRegressor.predict."""
        D = np.concatenate((X, Z, y.reshape(-1, 1)), axis=1)
        return super().predict(D, **kwargs)

    def _argmax_arm(self, dataset, x, f, requires_std=False):
        """Find the maximum value of f over the arms.  Uses batches to
        reduce the number of calls to self.predict()."""
        best_value, best_index = -np.inf, None
        for offs in range(0, len(dataset.arms), self.batch_size):

            batch = np.array(dataset.arms[offs:offs + self.batch_size],
                             dtype=object)
            X = np.vstack((x,) * len(batch))
            Z = np.vstack(tuple(batch[:, 0]))
            y = batch[:, 1].astype(float)

            if requires_std:
                mean, std = self.predict(X, Z, y, return_std=True)
                values = f(mean, std)
            else:
                mean = self.predict(X, Z, y, return_std=False)
                values = f(mean)

            index = np.argmax(values)
            if values[index] > best_value:
                best_value = values[index]
                best_index = index + offs

        return best_index

    def predict_arm(self, dataset, x):
        """Returns the best arm for context x."""
        j = self._argmax_arm(dataset,
                             np.array([x]),
                             lambda mean: mean,
                             requires_std=False)
        return dataset.arms[j]

    def _select_arm_at_random(self, dataset, x, beta):
        """Returns a random arm."""
        return dataset.arms[self.random_state.choice(len(dataset.arms))]

    def _select_arm_by_ucb(self, dataset, x, beta):
        """Returns the arm that maximizes the UCB of x."""
        j = self._argmax_arm(dataset,
                             np.array([x]),
                             lambda mean, std: mean + np.sqrt(beta) * std,
                             requires_std=True)
        return dataset.arms[j]

    def select_arm(self, dataset, x, beta=1.0):
        if self.strategy == 'random':
            return self._select_arm_at_random(dataset, x, beta)
        elif self.strategy == 'ucb':
            return self._select_arm_by_ucb(dataset, x, beta)
        else:
            raise ValueError('strategy must be "random" or "ucb"')
