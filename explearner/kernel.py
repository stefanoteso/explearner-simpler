import numpy as np
from sklearn.gaussian_process.kernels import CompoundKernel, GenericKernelMixin, Kernel, Hyperparameter

from explearner import kendall_tau_dist, kendall_tau_dist_vec


class CombinerKernel(GenericKernelMixin, CompoundKernel):
    """A kernel on triples."""
    def __init__(self, kx, kz, ky, nx, nz, combiner):
        self.kx, self.kz, self.ky = kx, kz, ky
        self.nx, self.nz = nx, nz
        self.combiner = combiner
        self.k1 = kx

        if combiner == 'sum':
            self._calc = self._calc_sum
            self._grad = self._grad_sum
        elif combiner == 'prod':
            self._calc = self._calc_prod
            self._grad = self._grad_prod
        else:
            raise ValueError(f'unknown combiner {combiner}')

        super().__init__((kx, kz, ky))

    def get_params(self, deep=True):
        return dict(kx=self.kx, kz=self.kz, ky=self.ky,
                    nx=self.nx, nz=self.nz, combiner=self.combiner)

    def _unpack(self, D):
        if D is None:
            return None, None, None
        X = D[:, :self.nx]
        Z = D[:, self.nx:self.nx+self.nz]
        Y = D[:, -1].reshape(-1, 1)
        return X, Z, Y

    @staticmethod
    def _calc_sum(Kx, Kz, Ky):
        return (Kx + Kz) * Ky

    @staticmethod
    def _grad_sum(Kx, Kz, Ky, Gx, Gz, Gy):
        Ky = Ky[:, :, np.newaxis]
        return np.dstack([
            Gx * Ky,
            Gz * Ky,
            (Kx + Kz)[:, :, np.newaxis] * Gy,
        ])

    @staticmethod
    def _calc_prod(Kx, Kz, Ky):
        return Kx * Kz * Ky

    @staticmethod
    def _grad_prod(Kx, Kz, Ky, Gx, Gy, Gz):
        Kx = Kx[:, :, np.newaxis]
        Kz = Kz[:, :, np.newaxis]
        Ky = Ky[:, :, np.newaxis]
        return np.dstack([
            Gx * Kz * Ky,
            Kx * Gz * Ky,
            Kx * Kz * Gy,
        ])

    def __call__(self, A, B=None, eval_gradient=False):
        AX, AZ, AY = self._unpack(A)
        BX, BZ, BY = self._unpack(B)
        if eval_gradient:
            Kx, Gx = self.kx(AX, BX, eval_gradient=True)
            Kz, Gz = self.kz(AZ, BZ, eval_gradient=True)
            Ky, Gy = self.ky(AY, BY, eval_gradient=True)
            return (self._calc(Kx, Kz, Ky),
                    self._grad(Kx, Kz, Ky, Gx, Gz, Gy))
        else:
            Kx = self.kx(AX, BX, eval_gradient=False)
            Kz = self.kz(AZ, BZ, eval_gradient=False)
            Ky = self.ky(AY, BY, eval_gradient=False)
            return self._calc(Kx, Kz, Ky)

    def diag(self, D):
        X, Z, Y = self._unpack(D)
        return self._calc(self.kx.diag(X),
                          self.kz.diag(Z),
                          self.ky.diag(Y))

class KendallKernel(GenericKernelMixin, Kernel):
    """
       Class for the Kendall Tau-like kernel, with:

            (C-D)/comb(n,2)

       where C,D is the number of concordant and discordant pairs, respectively.
    """
    def __init__(self, noise=1, noise_bounds=(1, 1)):
        self.noise = noise
        self.noise_bounds = noise_bounds

    @property
    def hyperparameter_noise(self):
        return Hyperparameter("noise", "numeric", self.noise_bounds)

    #TODO: Gradient?
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        n = X.shape[0]
        if Y is None:
            K = np.eye(n)
        else:
            K = 1 - kendall_tau_dist_vec(X, Y)
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
        if eval_gradient:
            return K, np.empty((n, n, 0))
        else:
            return K

    def diag(self, X):
        return np.ones(np.shape(X)[0])

    def is_stationary(self):
        return False
