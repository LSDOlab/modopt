# Test the visualization with recording and hot start functionalities

from all_problem_types import (SecondOrderScaling, second_order_scaling_lite, 
                                Scaling, scaling_lite, EqConstrained, eq_constrained_lite,
                                Feasibility, feasibility_lite, Unconstrained, unconstrained_lite,
                                IneqConstrained, ineq_constrained_lite)
from modopt import optimize, SLSQP, SNOPT, PySLSQP
from test_csdl import prob as csdl_prob
from test_csdl import alpha_prob
from test_openmdao import prob as om_prob
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import h5py

@pytest.mark.slsqp
@pytest.mark.visualization
@pytest.mark.slow
@pytest.mark.interfaces
def test_visualization():
    for prob in [Scaling(), scaling_lite()]:
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, visualize=['x[0]', 'obj', 'grad[1]', 'jac[1,0]', 'con[1]'])
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        print(results)
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
        assert_almost_equal(results['fun'], 20., decimal=6)
        assert prob._callback_count == 79
        assert prob._reused_callback_count == 0

@pytest.mark.visualization
@pytest.mark.slow
@pytest.mark.slsqp
@pytest.mark.recording
@pytest.mark.interfaces
def test_visualization_recording_hot_start():
    for prob in [Scaling(), scaling_lite()]:
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, recording=True, visualize=['x[0]', 'obj', 'grad[1]', 'jac[1,0]', 'con[1]'])
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        print(results)
        optimizer.print_results()
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
        assert_almost_equal(results['fun'], 20., decimal=6)
        assert prob._callback_count == 79
        assert prob._reused_callback_count == 0

        filename    = f'{optimizer.out_dir}/record.h5py'
        file        = h5py.File(filename, 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 79
        assert len(iterations) == 17

        # hot start from previous optimization
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, hot_start_from=filename, visualize=['x[0]', 'obj', 'grad[1]', 'jac[1,0]', 'con[1]'])
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        print(results)
        optimizer.print_results()
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
        assert_almost_equal(results['fun'], 20., decimal=6)
        assert prob._callback_count == 79
        assert prob._reused_callback_count == 79
        
        # hotstart and record from first optimization
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, hot_start_from=filename, recording=True, visualize=['x[0]', 'obj', 'grad[1]', 'jac[1,0]', 'con[1]'])
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        print(results)
        optimizer.print_results()
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
        assert_almost_equal(results['fun'], 20., decimal=6)
        assert prob._callback_count == 79
        assert prob._reused_callback_count == 79

        filename    = f'{optimizer.out_dir}/record.h5py'
        file        = h5py.File(filename, 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 79
        assert len(iterations) == 17

if __name__ == '__main__':
    test_visualization()
    test_visualization_recording_hot_start()
    print('All tests passed!')