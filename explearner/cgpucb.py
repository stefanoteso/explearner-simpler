import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor as GPR


class CGPUCB(GPR):
    def __init__(self, **kwargs):
        strategy = kwargs.pop('strategy', 'random')
        self.select_arm = {
            'random': self._select_arm_at_random,
            'ucb': self._select_arm_by_ucb,
        }[strategy]
        super().__init__(**kwargs)

    def fit(self, X, Z, y, f):
        """Wrapper around GaussianProcessRefressor.fit."""
        D = np.concatenate((X, Z, y.reshape(-1, 1)), axis=1)
        return super().fit(D, f)

    def predict(self, X, Z, y, **kwargs):
        """Wrapper around GaussianProcessRefressor.predict."""
        D = np.concatenate((X, Z, y.reshape(-1, 1)), axis=1)
        return super().predict(D, **kwargs)

    def predict_arm(self, dataset, x):
        """Returns the best arm for context x."""
        x = x[:, None]

        def mean_reward(arm):
            z, y = arm
            return self.predict(x, z[:, None], y[None, None])[0]

        return max(dataset.arms, key=mean_reward)

    def _select_arm_by_ucb(self, dataset, x, beta=1.0):
        """Returns the arm that maximizes the UCB of x."""
        x = x[:, None]

        def ucb(arm):
            z, y = arm
            mean, std = self.predict(x, z[:, None], y[None, None],
                                     return_std=True)
            return mean[0] + np.sqrt(beta) * std[0]

        return max(dataset.arms, key=ucb)

    def _select_arm_at_random(self, dataset, x, beta=1.0):
        """Returns a random arm."""
        return dataset.arms[self.random_state.choice(len(dataset.arms))]
