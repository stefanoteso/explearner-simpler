import numpy as np
from sklearn.gaussian_process.kernels import CompoundKernel, GenericKernelMixin, Kernel
from sklearn.base import clone


class CombinerKernel(GenericKernelMixin, CompoundKernel):
    """A kernel on triples."""
    def __init__(self, kx, kz, ky, nx, nz, combiner):
        self.kx, self.kz, self.ky = kx, kz, ky
        self.k1 = kx
        self.nx, self.nz = nx, nz
        self.combiner = combiner
        if combiner == 'sum':
            self._combine = lambda u, v, w: (u + v) * w
        elif combiner == 'prod':
            self._combine = lambda u, v, w: u * v * w
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

    def __call__(self, A, B=None, eval_gradient=False):
        AX, AZ, AY = self._unpack(A)
        BX, BZ, BY = self._unpack(B)
        if eval_gradient:
            Kx, Kx_gradient = self.kx(AX, BX, eval_gradient=True)
            Kz, Kz_gradient = self.kz(AZ, BZ, eval_gradient=True)
            Ky, Ky_gradient = self.ky(AY, BY, eval_gradient=True)
            K = self._combine(Kx, Kz, Ky)
            grad = np.dstack((Kx_gradient, Ky_gradient, Kz_gradient)) # XXX hack
            return K, grad
        else:
            Kx = self.kx(AX, BX, eval_gradient=False)
            Kz = self.kz(AZ, BZ, eval_gradient=False)
            Ky = self.ky(AY, BY, eval_gradient=False)
            return self._combine(Kx, Kz, Ky)

    def diag(self, D):
        X, Z, Y = self._unpack(D)
        return self._combine(self.kx.diag(X),
                             self.kz.diag(Z),
                             self.ky.diag(Y))

