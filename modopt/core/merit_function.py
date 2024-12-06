import numpy as np
from modopt.utils.options_dictionary import OptionsDictionary
from types import MethodType, FunctionType


# Note: Merit functions are only needed for constrained optimization
class MeritFunction(object):
    def __init__(self, **kwargs):
        self.options = OptionsDictionary()

        self.options.declare('f', types=(MethodType, FunctionType))
        self.options.declare('g', types=(MethodType, FunctionType))
        self.options.declare('c', types=(MethodType, FunctionType))
        self.options.declare('j', types=(MethodType, FunctionType))

        self.options.declare('nx',
                             types=(int, np.int32, np.int64, float, np.float64))
        self.options.declare('nc',
                             types=(int, np.int32, np.int64, float, np.float64))
        self.options.declare('nc_e', default=0,
                             types=(int, np.int32, np.int64, float, np.float64))

        self.initialize()

        self.options.update(kwargs)

        self.setup()

    def initialize(self):
        pass

    def setup(self):
        pass
