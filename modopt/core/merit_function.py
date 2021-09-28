import numpy as np
from modopt.utils.options_dictionary import OptionsDictionary


# Note: Merit functions are only needed for constrained optimization
class MeritFunction(object):
    def __init__(self, **kwargs):
        self.options = OptionsDictionary()

        self.options.declare('f', types=type(lambda: None))
        self.options.declare('g', types=type(lambda: None))
        self.options.declare('c', types=type(lambda: None))
        self.options.declare('j', types=type(lambda: None))

        self.initialize()

        self.options.update(kwargs)

    def initialize(self):
        pass
