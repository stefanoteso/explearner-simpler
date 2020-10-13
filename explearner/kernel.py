import numpy as np
from sklearn.gaussian_process.kernels import CompoundKernel, GenericKernelMixin
from sklearn.base import clone


class CombinerKernel(GenericKernelMixin, CompoundKernel):
    """A kernel on triples."""
    def __init__(self, kx, kz, ky, combiner):
        self.kx, self.kz, self.ky, self.combiner = kx, kz, ky, combiner
        if combiner == 'sum':
            self._combine = lambda u, v, w: (u + v) * w
        elif combiner == 'prod':
            self._combine = lambda u, v, w: u * v * w
        else:
            raise ValueError(f'unknown combiner {combiner}')
        super().__init__((kx, kz, ky))

    def get_params(self, deep=True):
        return dict(kx=self.kx, kz=self.kz, ky=self.ky, combiner=self.combiner)

    def __call__(self, A, B=None, eval_gradient=False):
        if eval_gradient:
            Kx, Kx_gradient = self.kx(A, B, eval_gradient=True)
            Kz, Kz_gradient = self.kz(A, B, eval_gradient=True)
            Ky, Ky_gradient = self.ky(A, B, eval_gradient=True)
            K = self._combine(Kx, Kz, Ky)
            grad = np.dstack((Kx_gradient, Ky_gradient, Kz_gradient)) # XXX hack
            return K, grad
        else:
            Kx = self.kx(A, B, eval_gradient=False)
            Kz = self.kz(A, B, eval_gradient=False)
            Ky = self.ky(A, B, eval_gradient=False)
            return self._combine(Kx, Kz, Ky)

    def diag(self, X):
        return self._combine(self.kx.diag(X),
                             self.kz.diag(X),
                             self.ky.diag(X))
