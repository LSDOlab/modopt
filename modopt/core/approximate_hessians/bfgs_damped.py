import numpy as np

from modopt import ApproximateHessian
from scipy.linalg import get_blas_funcs
import warnings
# from modopt_linalg import ldl, invert_upper_triangle

class BFGSDamped(ApproximateHessian):

    # Performs a rank 1 update of a symmetric matrix: call syr2(alpha, x, [a]); A := alpha*x*x' + A,
    # where A is an nxn symmetric matrix, x is an n-vector, and alpha is a real scalar.
    _syr = get_blas_funcs('syr', dtype='d') # dtype='d' means double precision

    # Performs a rank 2 update of a symmetric matrix: call syr2(alpha, x, y, [a]); A := alpha*x*y'+ alpha*y*x' + A,
    # where A is an nxn symmetric matrix, x and y are n-vectors, and alpha is a real scalar.
    # uplo specifies whether the upper or lower triangular part of A is stored.
    _syr2 = get_blas_funcs('syr2', dtype='d')

    # Performs a symmetric matrix-vector product: call symv(alpha, a, x); y := alpha*A*x + beta*y,
    # where A is an nxn symmetric matrix, x and y are n-vectors, alpha and beta are real scalars.
    _symv = get_blas_funcs('symv', dtype='d')

    def initialize(self):
        self.options.declare('store_hessian', default=True, types=bool)
        self.options.declare('store_inverse', default=False, types=bool)
        # The following option is only there for compatibility with the BFGSScipy class. It is not used in this class.
        self.options.declare('exception_strategy',
                             default='damp_update',
                             values=['skip_update', 'damp_update'])
        self.options.declare('min_curvature',
                             default=0.2,
                             types=float)
        self.options.declare('init_scale',
                             default='auto',
                             types=(float, str, np.ndarray))
        
    def setup(self, ):
        nx            = self.options['nx']
        min_curvature = self.options['min_curvature']
        init_scale    = self.options['init_scale']

        if min_curvature < 0.0:
            raise ValueError('min_curvature must be non-negative.')

        if isinstance(init_scale, float):
            if init_scale <= 0.0:
                raise ValueError('init_scale provided is not positive.')
        elif isinstance(init_scale, np.ndarray):
            if init_scale.size != nx and init_scale.size != nx**2: 
                raise ValueError(f'init_scale must be either the diagonal vector of size {nx} of a diagonal Hessian approximation ' \
                                 f'or a full Hessian approximation of size {nx}x{nx}')
            if init_scale.size == nx:
                if np.any(init_scale <= 0.0):
                    raise ValueError('init_scale diagonal vector contains non-positive elements.')
                init_scale = np.diag(init_scale)
            else:
                if np.any(init_scale != init_scale.T):
                    raise ValueError('init_scale full Hessian approximation is not symmetric.')
                if np.any(np.linalg.eigvals(init_scale) <= 0.0):
                    raise ValueError('init_scale Hessian approximation is not positive definite.')
        elif init_scale != 'auto':
            raise ValueError('Invalid value encountered for "init_scale". Must be a float, numpy.ndarray, or "auto".')
        else:
            # if init_scale == 'auto', set the initial scale to 1.0 for initialization with the identity matrix
            init_scale = 1.
        
        if self.options['store_hessian']:
            self.B_k   = np.eye(self.options['nx']) * init_scale

            # if isinstance(init_scale, np.ndarray) and init_scale.size == nx**2:
            #     # Implement Cholesky factorization of the initial Hessian approximation
            #     L          = np.linalg.cholesky(self.B_k)
            #     self.R     = L.T
            #     self.R_inv = invert_upper_triangle(self.R)
            #     diag       = np.diag(L)
            #     if any(diag <= 0.0):
            #         raise ValueError('init_scale full Hessian approximation is not positive definite.')
            #     self.L     = L / diag  # rowwise division
            #     np.fill_diagonal(self.L, diag**2)

            #     raise warnings.warn("Full Hessian approximation's LDL factorization is not tested.")

            # else:
            #     self.L     = np.eye(nx) * init_scale
            #     self.R     = np.eye(nx) * (init_scale ** 0.5)
            #     self.R_inv = np.diag(1. / np.diag(self.R)) # Note that init_scale here is a scalar or a diagonal matrix

        if self.options['store_inverse']:
            self.M_k = np.eye(self.options['nx']) * init_scale

        self.is_first_iteration = True

    def update(self, d, w):

        if np.all(d == 0.):
            warnings.warn('Direction vector is zero. Skipping update.')
            return
        if np.all(w == 0.):
            warnings.warn('Gradient delta vector is zero. Skipping update.')
            return
        if np.any(np.isnan(d)):
            warnings.warn('Direction vector contains NaNs. Skipping update.')
            return
        if np.any(np.isnan(w)):
            warnings.warn('Gradient delta vector contains NaNs. Skipping update.')

        mincurv = self.options['min_curvature']
        
        wTd = np.dot(w, d)
        wTw = np.dot(w, w)
        dTd = np.dot(d, d)

        # wTd == 0 already includes the case where wTw or dTd is zero
        auto_scale = 1. if (wTd == 0.) else np.abs(wTw/wTd)
        # The following alternative does not work as well as the above line.
        # Numerical experimentation suggests that it is better to scale even if wTd is negative using abs(wTd).
        # auto_scale = 1. if (wTd <= 0.) else wTw / wTd

        if self.options['store_hessian']:

            if self.is_first_iteration:
                if self.options['init_scale'] == 'auto':
                    self.B_k = auto_scale * np.eye(self.options['nx'])
                self.is_first_iteration = False

            Bd  = self._symv(1, self.B_k, d) # More efficient than: Bd = np.dot(self.B_k, d)
            dBd = np.dot(d, Bd)

            # Check if positive definiteness is violated due to rounding errors. 
            # Ideally, it should not happen in exact arithmetic, but is bound to happen in finite precision arithmetic.
            # Reinitialize the Hessian if it is violated
            if dBd <= 0:
            # if dBd/dTd < 1e-12: # extremely small eigenvalues in the Hessian approximation, includes the case where dBd <= 0 
                self.B_k = auto_scale * np.eye(self.options['nx'])
                # Recalculate Bd and dBd
                Bd  = self._symv(1, self.B_k, d)
                dBd = np.dot(d, Bd)


            # Damped BFGS update
            wTd_min = mincurv * dBd
            if wTd < wTd_min:
                # theta = (1 - mincurv) * dBd / (dBd-wTd)
                theta = (1 - mincurv) / (1-wTd/dBd)
                w     = theta*w + (1-theta)*Bd
                # wTd   = wTd_min * 1.
                wTd   = np.dot(w, d)

            # Perform the BFGS update
            # self.B_k += np.outer(w, w) / wTd - np.outer(Bd, Bd) / dBd

            # Using BLAS routines to perform the update since they are faster for these operations
            self.B_k  = self._syr( 1./wTd,  w, a=self.B_k, lower=0)
            self.B_k  = self._syr(-1./dBd, Bd, a=self.B_k, lower=0)

            # Ensure symmetry of the Hessian approximation as blas only updates the upper triangular part
            lower_indices = np.tril_indices_from(self.B_k, -1)
            self.B_k[lower_indices] = self.B_k.T[lower_indices]

            # # Maintaining LDL factorization of the BFGS Hessian approximation
            # self.L = ldl(self.L,  w,  1/wTd)
            # self.L = ldl(self.L, Bd, -1/dBd)

            # D = np.diag(np.diag(self.L))
            # L = self.L * 1.
            # np.fill_diagonal(L, 1.)
            # self.B_k = L @ D @ L.T
            # print('BFGS condition number:', max(np.diagonal(D))/min(np.diagonal(D)))

            # self.R = (D ** 0.5) @ L.T
            # # print('error in R', np.linalg.norm(self.B-self.R.T@self.R))
            
            # self.R_inv = invert_upper_triangle(self.R)

        if self.options['store_inverse']:
            # Perform a simple update with no damping or resets
            Mw  = np.dot(self.M_k, w)
            wMw = np.dot(w, Mw)
            wTd = np.dot(w, d)

            # Broyden-one-parameter family type formula of performing the update
            # vec = d / wTd - Mw / wMw
            # self.M_k += np.outer(d, d) / wTd - np.outer(Mw, Mw) / wMw  + wMw * np.outer(vec, vec)

            # Another equivalent way to write the above formula
            # I   = np.eye(len(d))
            # rho = 1. / wTd if wTd != 0 else 1000.
            # self.M_k  = (I - rho*np.outer(d, w)) @ self.M_k @ (I - rho*np.outer(w, d)) + rho*np.outer(d, d)

            # A third alternative to write the above formulas
            # self.M_k += np.outer(d, d) * (wTd + wMw) / (wTd**2) - 1/wTd * (np.outer(Mw, d) + np.outer(d, Mw))
            self.M_k  = self._syr((wTd + wMw) / (wTd**2),  d, a=self.M_k)
            self.M_k  = self._syr2( -1./wTd,  d, Mw, a=self.M_k)