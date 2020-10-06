import numpy as np

from sklearn.utils import check_random_state


class CGPUCB:
   # TODO convert to sklearn GaussianProcessRegressor

    def __init__(self, dataset, strategy='random', rng=None):
        self.rng = check_random_state(rng)
        self.select_arm = {
            'random': self._select_arm_at_random,
            'ucb': self._select_arm_by_ucb,
        }[strategy]

        D = self.concat(dataset.X, dataset.Z, dataset.y.reshape(-1, 1))
        super().__init__((D, dataset.f), dataset.kernel)

    @staticmethod
    def concat(X, Z, Y):
        return np.concatenate((X, Z, Y), axis=1)

    def predict_reward(self, x, z, y):
        """Computes the mean reward and std. dev. for (x, z, y)."""
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
        y = np.array([[y]])
        mean, var = super().predict_y(self.concat(x, z, y))
        return mean.numpy(), var.numpy()

    def predict(self, dataset, x):
        """Returns the best arm for context x."""
        return max(dataset.arms, key=lambda arm: self.predict_reward(x, *arm))

    def ucb(self, x, z, y, beta_t=1):
        """Computes the upper confidence bound of (x, z, y)."""
        mean, var = self.predict_reward(x, z, y)
        return (mean + np.sqrt(beta_t) * var)[0][0]

    def _select_arm_at_random(self, dataset, x):
        """Returns a random arm."""
        return dataset.arms[self.rng.choice(len(dataset.arms))]

    def _select_arm_by_ucb(self, dataset, x):
        """Returns the arm that maximizes the UCB of x."""
        return max(dataset.arms, key=lambda arm: self.ucb(x, *arm))
