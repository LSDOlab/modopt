import numpy as np

from modopt.utils.options_dictionary import OptionsDictionary


class ApproximateHessian(object):
    def __init__(self, **kwargs):
        self.options = OptionsDictionary()

        self.options.declare('nx', types=(int, np.int32, np.int64))
        self.initialize()

        self.options.update(kwargs)

        if 'store_hessian' in self.options:
            if self.options['store_hessian']:
                self.B_k = np.identity(self.options['nx'])

        if 'store_inverse' in self.options:
            if self.options['store_inverse']:
                self.M_k = np.identity(self.options['nx'])

        self.setup()

    def setup(self,):
        pass