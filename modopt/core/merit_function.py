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

        self.cache = {}
        self.eval_count = {'f': 0, 'c': 0, 'g': 0, 'j': 0}

        self.initialize()

        self.options.update(kwargs)

        self.setup()

    def clear_cache(self):
        """
        Clear the cache of function evaluations.
        """
        self.cache = {}

    def initialize(self):
        pass

    def setup(self):
        pass

    def update_functions_in_cache(self, fnames, x):
        """
        Update the function values in the cache for the specified functions 
        in `fnames` with their values at the given point `x`.
        If the cache contains no values for a function in `fnames` 
        or the cached values correspond to a point other than `x`, 
        the function is evaluated at `x` and the cache is updated accordingly.

        Parameters
        ----------
        fnames : str or list of str
            Function names whose values need to be updated in the cache.
            `fnames` must be one or a subset of ['f', 'g', 'c', 'j'].
        x : np.ndarray
            Point at which to update the functions in `fnames`.
        """

        if not isinstance(fnames, (list, str)):
            raise ValueError("'fnames' must be a string or list of strings")
        
        fnames = [fnames] if isinstance(fnames, str) else fnames
        possible_fnames = ['f', 'g', 'c', 'j']
        if any([fname not in possible_fnames for fname in fnames]):
            raise ValueError("Invalid function name. Must be one of: ", possible_fnames)
        
        if not isinstance(x, np.ndarray):
            raise ValueError("'x' must be a numpy array")
        
        for fname in fnames:
            if fname not in self.cache:
                self.cache[fname] = (x, self.options[fname](x))
                self.eval_count[fname] += 1
            elif not np.array_equal(x, self.cache[fname][0]):
                self.cache[fname] = (x, self.options[fname](x))
                self.eval_count[fname] += 1
            # else:
            #     print("Cache hit for function: ", fname)
