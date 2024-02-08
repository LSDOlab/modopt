
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
