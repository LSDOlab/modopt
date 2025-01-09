# Test the recording and hot start functionalities

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
@pytest.mark.recording
def test_recording():
    
    for prob in [Scaling(), scaling_lite()]:
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, recording=True)
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        print(results)
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
        assert_almost_equal(results['fun'], 20., decimal=6)
        assert prob._callback_count == 79
        assert prob._reused_callback_count == 0

        file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 79
        assert len(iterations) == 17

@pytest.mark.csdl
@pytest.mark.csdl_alpha
@pytest.mark.recording
@pytest.mark.interfaces
def test_csdl_recording():
    from test_csdl import prob as csdl_prob
    from test_csdl import alpha_prob

    for prob in [csdl_prob, alpha_prob]:
        optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True}, recording=True)
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        print(results)
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_almost_equal(results['fun'], 3., decimal=11)
        assert prob._callback_count == 25
        assert prob._reused_callback_count == 0

        file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 25
        assert len(iterations) == 5

@pytest.mark.openmdao
@pytest.mark.recording
@pytest.mark.interfaces
def test_openmdao_recording():
    from test_openmdao import prob

    optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True}, recording=True)
    optimizer.check_first_derivatives()
    results = optimizer.solve()
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_almost_equal(optimizer.results['x'], [200., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 6., decimal=11)
    assert prob._callback_count == 25
    assert prob._reused_callback_count == 0

    file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
    groups      = list(file.keys())
    callbacks   = [key for key in groups if key.startswith('callback_')]
    iterations  = [key for key in groups if key.startswith('iteration_')]
    assert len(callbacks)  == 25
    assert len(iterations) == 5

@pytest.mark.csdl
@pytest.mark.csdl_alpha
@pytest.mark.openmdao
@pytest.mark.recording
@pytest.mark.interfaces
@pytest.mark.snopt
def test_compute_all_recording():
    from test_csdl import prob as csdl_prob
    from test_csdl import alpha_prob
    from test_openmdao import prob as om_prob

    for prob in [csdl_prob, alpha_prob, om_prob]:
        optimizer = SNOPT(prob, recording=True)
        results = optimizer.solve()
        print(results)
        assert results['info'] == 1
        assert prob._callback_count == 6
        assert prob._reused_callback_count == 0

        file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 6
        assert len(iterations) == 0

@pytest.mark.interfaces
@pytest.mark.pyslsqp
@pytest.mark.cobyla
@pytest.mark.cobyqa
@pytest.mark.lbfgsb
@pytest.mark.trustconstr
@pytest.mark.bfgs
@pytest.mark.neldermead
@pytest.mark.sqp
@pytest.mark.ipopt
@pytest.mark.cvxopt
@pytest.mark.qpsolvers
def test_all_solvers_recording(): # except SNOPT and SLSQP (already tested above)
    from modopt import PySLSQP, COBYLA, COBYQA, LBFGSB, TrustConstr, BFGS, NelderMead, SQP, IPOPT, CVXOPT, ConvexQPSolvers
    from all_problem_types import Unconstrained, BoundConstrained, IneqConstrained, Scaling, ConvexQP
    from test_cvxopt import ConstrainedBoundedConvex

    optimizer = BFGS(Unconstrained(), recording=True)
    results = optimizer.solve()
    print(results)

    file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
    groups      = list(file.keys())
    callbacks   = [key for key in groups if key.startswith('callback_')]
    iterations  = [key for key in groups if key.startswith('iteration_')]
    assert len(callbacks)  == 143
    assert len(iterations) == 67

    for OPT in [LBFGSB, NelderMead]:
        optimizer = OPT(BoundConstrained(), recording=True)
        results = optimizer.solve()
        print(results)

        file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]

    optimizer = COBYLA(IneqConstrained(), recording=True)
    results = optimizer.solve()
    print(results)

    file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
    groups      = list(file.keys())
    callbacks   = [key for key in groups if key.startswith('callback_')]
    iterations  = [key for key in groups if key.startswith('iteration_')]
    assert len(callbacks)  == 179
    assert len(iterations) == 90

    for OPT in [PySLSQP, COBYQA, TrustConstr, SQP, IPOPT]:
        optimizer = OPT(Scaling(), recording=True)
        results = optimizer.solve()
        print(results)

        file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]

    optimizer = ConvexQPSolvers(ConvexQP(), solver_options={'solver':'osqp'}, recording=True)
    results = optimizer.solve()
    print(results)

    file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
    groups      = list(file.keys())
    callbacks   = [key for key in groups if key.startswith('callback_')]
    iterations  = [key for key in groups if key.startswith('iteration_')]
    assert len(callbacks)  == 6
    assert len(iterations) == 0

    optimizer = CVXOPT(ConstrainedBoundedConvex(), recording=True)
    results = optimizer.solve()
    print(results)

    file        = h5py.File(results['out_dir']+'/record.hdf5', 'r')
    groups      = list(file.keys())
    callbacks   = [key for key in groups if key.startswith('callback_')]
    iterations  = [key for key in groups if key.startswith('iteration_')]
    assert len(callbacks)  == 831
    assert len(iterations) == 0

@pytest.mark.slsqp
@pytest.mark.recording
def test_hot_start():
    for prob in [Scaling(), scaling_lite()]:
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, recording=True)
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

        filename    = results['out_dir']+'/record.hdf5'
        file        = h5py.File(filename, 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 79
        assert len(iterations) == 17

        # hot start from previous optimization
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, hot_start_from=filename)
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
        optimizer = SLSQP(prob, solver_options={'maxiter':50}, hot_start_from=filename, recording=True)
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

        filename    = results['out_dir']+'/record.hdf5'
        file        = h5py.File(filename, 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 79
        assert len(iterations) == 17

@pytest.mark.csdl
@pytest.mark.csdl_alpha
@pytest.mark.recording
@pytest.mark.interfaces
def test_csdl_hot_start():
    from test_csdl import prob as csdl_prob
    from test_csdl import alpha_prob

    for prob in [csdl_prob, alpha_prob]:
        optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True}, recording=True)
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_almost_equal(results['fun'], 3., decimal=11)
        assert prob._callback_count == 25
        assert prob._reused_callback_count == 0

        filename    = results['out_dir']+'/record.hdf5'
        file        = h5py.File(filename, 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 25
        assert len(iterations) == 5

        # hot start from previous optimization
        optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True}, hot_start_from=filename)
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_almost_equal(results['fun'], 3., decimal=11)
        assert prob._callback_count == 25
        assert prob._reused_callback_count == 25

        # hotstart and record from first optimization
        optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True}, hot_start_from=filename, recording=True)
        optimizer.check_first_derivatives()
        results = optimizer.solve()
        assert results['success'] == True
        assert results['message'] == 'Optimization terminated successfully'
        assert_almost_equal(results['fun'], 3., decimal=11)
        assert prob._callback_count == 25
        assert prob._reused_callback_count == 25

        filename    = results['out_dir']+'/record.hdf5'
        file        = h5py.File(filename, 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 25


@pytest.mark.openmdao
@pytest.mark.recording
@pytest.mark.interfaces
def test_openmdao_hot_start():
    from test_openmdao import prob

    optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True}, recording=True)
    optimizer.check_first_derivatives()
    results = optimizer.solve()
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_almost_equal(optimizer.results['x'], [200., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 6., decimal=11)
    assert prob._callback_count == 25
    assert prob._reused_callback_count == 0

    filename    = results['out_dir']+'/record.hdf5'
    file        = h5py.File(filename, 'r')
    groups      = list(file.keys())
    callbacks   = [key for key in groups if key.startswith('callback_')]
    iterations  = [key for key in groups if key.startswith('iteration_')]
    assert len(callbacks)  == 25
    assert len(iterations) == 5

    # hot start from previous optimization
    optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True}, hot_start_from=filename)
    optimizer.check_first_derivatives()
    results = optimizer.solve()
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_almost_equal(results['fun'], 6., decimal=11)
    assert prob._callback_count == 25
    assert prob._reused_callback_count == 25

    # hotstart and record from first optimization
    optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True}, hot_start_from=filename, recording=True)
    optimizer.check_first_derivatives()
    results = optimizer.solve()
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_almost_equal(results['fun'], 6., decimal=11)
    assert prob._callback_count == 25
    assert prob._reused_callback_count == 25

    filename    = results['out_dir']+'/record.hdf5'
    file        = h5py.File(filename, 'r')
    groups      = list(file.keys())
    callbacks   = [key for key in groups if key.startswith('callback_')]
    iterations  = [key for key in groups if key.startswith('iteration_')]
    assert len(callbacks)  == 25
    assert len(iterations) == 5

@pytest.mark.csdl
@pytest.mark.csdl_alpha
@pytest.mark.openmdao
@pytest.mark.recording
@pytest.mark.interfaces
@pytest.mark.snopt
def test_compute_all_hot_start():
    from test_csdl import prob as csdl_prob
    from test_csdl import alpha_prob
    from test_openmdao import prob as om_prob

    for prob in [csdl_prob, alpha_prob, om_prob]:
        optimizer = SNOPT(prob, recording=True)
        results = optimizer.solve()
        assert results['info'] == 1
        assert prob._callback_count == 6
        assert prob._reused_callback_count == 0
        
        filename    = results['out_dir']+'/record.hdf5'
        file        = h5py.File(filename, 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 6
        assert len(iterations) == 0

        # hot start from previous optimization
        optimizer = SNOPT(prob, hot_start_from=filename)
        results = optimizer.solve()
        assert results['info'] == 1
        assert prob._callback_count == 6
        assert prob._reused_callback_count == 6

        # hotstart and record from first optimization
        optimizer = SNOPT(prob, hot_start_from=filename, recording=True)
        results = optimizer.solve()
        assert results['info'] == 1
        assert prob._callback_count == 6
        assert prob._reused_callback_count == 6

        filename    = results['out_dir']+'/record.hdf5'
        file        = h5py.File(filename, 'r')
        groups      = list(file.keys())
        callbacks   = [key for key in groups if key.startswith('callback_')]
        iterations  = [key for key in groups if key.startswith('iteration_')]
        assert len(callbacks)  == 6
        assert len(iterations) == 0

@pytest.mark.slsqp
@pytest.mark.recording
def test_errors():
    prob    = Scaling()
    with pytest.raises(ValueError) as exc_info:
        optimizer = SLSQP(prob, recording=True, turn_off_outputs=True)
    assert str(exc_info.value) == "Cannot record with 'turn_off_outputs=True'."
    
    with pytest.raises(ValueError) as exc_info:
        optimizer = SLSQP(prob, readable_outputs=['x'], turn_off_outputs=True)
    assert str(exc_info.value) == "Cannot write 'readable_outputs' with 'turn_off_outputs=True'."

if __name__ == '__main__':
    test_recording()
    test_csdl_recording()
    test_openmdao_recording()
    test_compute_all_recording()
    test_all_solvers_recording()
    test_hot_start()
    test_csdl_hot_start()
    test_openmdao_hot_start()
    test_compute_all_hot_start()
    test_errors()
    print('All tests passed!')