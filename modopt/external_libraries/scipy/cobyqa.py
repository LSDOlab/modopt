import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer

class COBYQA(Optimizer):
    pass