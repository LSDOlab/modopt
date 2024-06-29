
# From OpenMDAO (for printing check_first_derivatives())
def pad_name(name, pad_num=10, quotes=False):
    """
    Pad a string so that they all line up when stacked.
    Parameters
    ----------
    name : str
        The string to pad.
    pad_num : int
        The number of total spaces the string should take up.
    quotes : bool
        If name should be quoted.
    Returns
    -------
    str
        Padded string.
    """
    l_name = len(name)
    quotes_len = 2 if quotes else 0
    if l_name + quotes_len < pad_num:
        pad = pad_num - (l_name + quotes_len)
        if quotes:
            pad_str = "'{name}'{sep:<{pad}}"
        else:
            pad_str = "{name}{sep:<{pad}}"
        pad_name = pad_str.format(name=name, sep='', pad=pad)
        return pad_name
    else:
        if quotes:
            return "'{0}'".format(name)
        else:
            return '{0}'.format(name)
        

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Usage:
# ======
# with HiddenPrints():
#     print("This will not be printed")

# print("This will be printed as before")

class ImmutableKeysDict(dict):
    '''
    Allows modifications to existing keys but will raise a KeyError 
    if there's an attempt to add a new key.
    '''
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"Cannot add new key: {key}")
        super().__setitem__(key, value)

# Usage: Initialize with default values
# ======
# default_solver_options = ImmutableKeysDict({
#     'show_progress': True,
#     'maxiters': 100,
#     'abstol': 1E-6,
#     'reltol': 1E-6,
#     'feastol': 1E-6,
#     'abstol_inacc': 1E-6,
#     'reltol_inacc': 1E-6,
#     'feastol_inacc': 1E-6
# })