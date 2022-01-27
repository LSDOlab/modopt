import numpy as np
from types import FunctionType, MethodType

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt.utils.options_dictionary import OptionsDictionary


class QPSolver(object):
    def __init__(self, **kwargs):
        self.options = OptionsDictionary()

        self.options.declare('nx',
                             types=(int, np.int32, np.int64, float,
                                    np.float64))
        self.options.declare('nc',
                             types=(int, np.int32, np.int64, float,
                                    np.float64))
        self.options.update(kwargs)

        nx = self.options['nx']
        nc = self.options['nc']

        # Objective: qTx + (1/2)xTPx
        self.options.declare('P',
                             types=np.ndarray,
                             default=np.zeros((nx, nx)))
        self.options.declare('q',
                             types=np.ndarray,
                             default=np.zeros((nx, )))

        # Constraints: lc <= Ax <= uc
        self.options.declare('A',
                             types=np.ndarray,
                             default=np.zeros((nc, nx)))
        self.options.declare('lc',
                             types=np.ndarray,
                             default=np.full((nc, ), -np.inf))
        self.options.declare('uc',
                             types=np.ndarray,
                             default=np.full((nc, ), np.inf))

        # Constraints: lb <= x <= ub
        self.options.declare('lb',
                             types=np.ndarray,
                             default=np.full((nx, ), -np.inf))
        self.options.declare('ub',
                             types=np.ndarray,
                             default=np.full((nx, ), np.inf))

        self.initialize()
        self.options.update(kwargs)
