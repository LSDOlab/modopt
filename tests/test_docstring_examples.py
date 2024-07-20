'''
This file contains tests for the examples within docstrings.
'''
import os
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(here, '../modopt/core'))  # Add the pyslsqp directory to the Python path
import doctest

import pytest

@pytest.mark.slsqp
@pytest.mark.interfaces
@pytest.mark.recording
def test_postprocessing():
    import postprocessing
    failures, _ = doctest.testmod(postprocessing)
    assert failures == 0, 'One or more doctests failed in main.py'


@pytest.mark.slsqp
@pytest.mark.interfaces
@pytest.mark.recording
@pytest.mark.visualization
def test_visualization():
    import visualization
    failures, _ = doctest.testmod(visualization)
    assert failures == 0, 'One or more doctests failed in visualize.py'


if __name__ == '__main__':
    test_postprocessing()
    test_visualization()