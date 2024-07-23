# Test Optimizer base class
import numpy as np
import pytest
from all_problem_types import Scaling, scaling_lite
from modopt import SLSQP, PySLSQP, SNOPT, IPOPT
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import os

@pytest.mark.interfaces
@pytest.mark.slsqp
@pytest.mark.ipopt
@pytest.mark.pyslsqp
def test_turn_off_outputs():
    for prob in [Scaling(), scaling_lite()]:
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, turn_off_outputs=True)
        assert not hasattr(optimizer, 'out_dir')
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
        assert_almost_equal(results['fun'], 20., decimal=6)

        # Test output filepaths for PySLSQP, IPOPT, and SNOPT
        optimizer = PySLSQP(prob, solver_options={'maxiter':50}, turn_off_outputs=True)
        assert not hasattr(optimizer, 'out_dir')
        results = optimizer.solve()
        assert results['success'] == True
        assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
        assert_almost_equal(results['objective'], 20., decimal=6)
        assert os.path.exists('slsqp_summary.out')

        optimizer = IPOPT(prob, turn_off_outputs=True)
        assert not hasattr(optimizer, 'out_dir')
        results = optimizer.solve()
        assert_array_almost_equal(results['x'], [2., 0.], decimal=9)
        assert_almost_equal(results['f'], 20., decimal=7)
        assert_almost_equal(results['c'], [5, 0.5], decimal=9)
        assert os.path.exists('ipopt_output.txt')

@pytest.mark.interfaces
@pytest.mark.snopt
def test_turn_off_outputs_snopt():
    for prob in [Scaling(), scaling_lite()]:
        optimizer = SNOPT(prob, solver_options={'Major optimality':1e-8}, turn_off_outputs=True)
        assert not hasattr(optimizer, 'out_dir')
        results = optimizer.solve()
        assert optimizer.results['info'] == 1
        assert_array_almost_equal(results['x'], [2., 0.], decimal=11)
        assert_almost_equal(results['objective'], 20., decimal=11)
        assert os.path.exists('SNOPT_summary.out')
        assert os.path.exists('SNOPT_print.out')

@pytest.mark.interfaces
@pytest.mark.slsqp
def test_callback_counts():
    for prob in [Scaling(), scaling_lite()]:
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, turn_off_outputs=True)
        assert not hasattr(optimizer, 'out_dir')
        results = optimizer.solve()
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
        assert_almost_equal(results['fun'], 20., decimal=6)

        optimizer.print_callback_counts()
        assert optimizer.problem._callback_count         == 71
        assert optimizer.problem._reused_callback_count  == 0
        assert optimizer.problem._obj_count              == 19
        assert optimizer.problem._grad_count             == 16
        assert optimizer.problem._hess_count             == 0
        assert optimizer.problem._con_count              == 20
        assert optimizer.problem._jac_count              == 16

if __name__ == "__main__":
    test_turn_off_outputs()
    test_turn_off_outputs_snopt()
    test_callback_counts()
    print("All tests passed!")