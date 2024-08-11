# Test the pycutest interface

from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

@pytest.mark.interfaces
@pytest.mark.pycutest
def test_pycutest():
    import numpy as np
    import pycutest as pc
    from modopt import SLSQP
    from modopt import CUTEstProblem
    from modopt.external_libraries.pycutest import find_problems, problem_properties, print_available_sif_params, import_problem

    # 1. Test mo.external_libraries.pycutest.find_problems
    pycutest_output = pc.find_problems(constraints='unconstrained', userN=True)
    modopt_output   =    find_problems(constraints='unconstrained', userN=True)
    assert pycutest_output == modopt_output

    # 2. Test mo.external_libraries.pycutest.problem_properties
    pycutest_output = pc.problem_properties('ROSENBR')
    modopt_output   =    problem_properties('ROSENBR')
    assert pycutest_output == modopt_output

    # 3. Test mo.external_libraries.pycutest.print_available_sif_params
    import io
    import contextlib

    # Capture the output of pc.print_available_sif_params
    pc_output_buffer = io.StringIO()
    with contextlib.redirect_stdout(pc_output_buffer):
        pc.print_available_sif_params('ARGLALE')
    pycutest_output = pc_output_buffer.getvalue()

    # Capture the output of print_available_sif_params
    modopt_output_buffer = io.StringIO()
    with contextlib.redirect_stdout(modopt_output_buffer):
        print_available_sif_params('ARGLALE')
    modopt_output = modopt_output_buffer.getvalue()

    # Assert that the outputs are the same
    assert pycutest_output == modopt_output

    pycutest_output = pc.import_problem('ARGLALE', sifParams={'N':100, 'M':200})
    modopt_output   =    import_problem('ARGLALE', sifParams={'N':100, 'M':200})
    # print(type(pycutest_output), type(modopt_output))
    # print(pycutest_output, modopt_output)
    assert pycutest_output == modopt_output

    # 4. Test mo.CUTEstProblem methods for an unconstrained problem

    # Test a second-order unconstrained problem
    print('Testing a second-order unconstrained problem: ROSENBR')
    print('Problem properties:', pc.problem_properties('ROSENBR'))
    prob_uc = CUTEstProblem(cutest_problem=pc.import_problem('ROSENBR'))
    optimizer = SLSQP(prob_uc, solver_options={'maxiter':100})
    optimizer.solve()
    assert optimizer.results['success']
    assert_array_almost_equal(optimizer.results['x'], [1., 1.], decimal=4)
    assert_almost_equal(optimizer.results['fun'], 0.0, decimal=8)

    obj_hess = prob_uc._compute_objective_hessian(np.array([1., 1.]))
    assert_array_equal(obj_hess, np.array([[802., -400.], [-400., 200.]]))
    obj_hvp  = prob_uc._compute_objective_hvp(np.array([1., 1.]), np.array([1., 1.]))
    assert_array_equal(obj_hvp, np.array([402., -200.]))

    with pytest.raises(ValueError) as excinfo:
        prob_uc._compute_lagrangian_hvp(np.ones(2), np.ones(2), 0)
    assert str(excinfo.value) == "Lagrangian Hessian-vector product is not defined for unconstrained CUTEST problems."\
                                 "Use 'compute_objective_hvp' for unconstrained problems."

    # 5. Test mo.CUTEstProblem methods for a constrained problem

    # probs = pc.find_problems(objective='quadratic', constraints='quadratic', userN=True)
    # print(probs)
    # for prob in probs:
    #     print(pc.problem_properties(prob))
    #     pc.print_available_sif_params(prob)

    # Test a second-order constrained problem
    print('Testing a second-order constrained problem: EIGENA2')
    print('Problem properties:', pc.problem_properties('EIGENA2'))
    prob_c = CUTEstProblem(cutest_problem=pc.import_problem('EIGENA2', sifParams={'N':10,}))
    assert prob_c.nx == 110
    assert prob_c.nc ==  55
    optimizer = SLSQP(prob_c, solver_options={'maxiter':100})
    optimizer.solve()
    # print(optimizer.results)
    assert optimizer.results['success']
    assert_almost_equal(optimizer.results['fun'], 0.0, decimal=10)

    obj_hess = prob_c._compute_objective_hessian(np.ones(110))
    with pytest.raises(ValueError) as excinfo:
        prob_c._compute_objective_hvp(np.ones(110), np.ones(110))
    assert str(excinfo.value) == "Objective Hessian-vector product is not defined for constrained CUTEST problems."\
                                 "Use 'compute_lagrangian_hvp' for constrained problems."
    
    lag_hess = prob_c._compute_lagrangian_hessian(np.ones(110), np.ones(55))

    lag_hvp = prob_c._compute_lagrangian_hvp(np.ones(110), np.ones(55), np.ones(110))
    assert_array_equal(lag_hvp, lag_hess @ np.ones(110))

    jac = prob_c._compute_constraint_jacobian(np.ones(110))

    jvp = prob_c._compute_constraint_jvp(np.ones(110), np.ones(110))
    assert_array_equal(jvp, jac @ np.ones(110))
    assert_array_equal(jvp, 20 * np.ones(55))

    vjp = prob_c._compute_constraint_vjp(np.ones(110), np.ones(55))
    assert_array_equal(vjp, np.ones(55) @ jac)
    
if __name__ == '__main__':
    test_pycutest()
    print('All tests passed!')