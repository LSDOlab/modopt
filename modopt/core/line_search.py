import numpy as np
from types import FunctionType, MethodType

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt.utils.options_dictionary import OptionsDictionary


class LineSearch(object):
    def __init__(self, **kwargs):
        self.options = OptionsDictionary()

        # Armijo parameter
        self.options.declare('eta_a',
                             default=1e-4,
                             types=float,
                             upper=(0.5 - eps),
                             lower=eps)

        # TODO: Pass in a merit function object

        self.options.declare('f', types=(FunctionType, MethodType))
        self.options.declare('g', types=(FunctionType, MethodType))

        self.initialize()
        self.options.update(kwargs)
