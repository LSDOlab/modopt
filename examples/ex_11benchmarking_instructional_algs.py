'''Benchmark instructional algorithms on three analytical problems'''

import numpy as np
from modopt import ProblemLite
import time
import matplotlib.pyplot as plt

# Problem 1: minimize sum(x^2)
x0 = np.array([1., 1.])
name = 'quartic_function'
obj  = lambda x: np.sum(x**4)
grad = lambda x: 4 * x**3
obj_hess = lambda x: 12 * np.diag(x**2)
prob1 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess)
sol1  = np.array([0., 0.])

# Problem 2: minimize (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
x0 = np.array([-1.2, 1.])
name = 'rosenbrock_function'
obj  = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
grad = lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
obj_hess = lambda x: np.array([[2 + 800*x[0]**2 - 400*(x[1]-x[0]**2), -400*x[0]], [-400*x[0], 200]])
prob2 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess)
sol2  = np.array([1., 1.])

# Problem 3: minimize (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1] - x[0]**2)**2
x0 = np.array([0., 0.])
name = 'bean_function'
obj  = lambda x: (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1] - x[0]**2)**2
grad = lambda x: np.array([-2*(1-x[0]) - 2*x[0]*(2*x[1] - x[0]**2), -2*(1-x[1]) + 2*(2*x[1] - x[0]**2)])
obj_hess = lambda x: np.array([[2 + 4*x[0]**2 - 2*(2*x[1] - x[0]**2), -4*x[0]], [-4*x[0], 2 + 4]])
prob3 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess)
sol3  = np.array([1.21314, 0.82414])

# Benchmarking optimization algorithms
from modopt import SteepestDescent, Newton, QuasiNewton
from modopt.core.optimization_algorithms.steepest_descent_no_ls import SteepestDescentNoLS
from modopt.core.optimization_algorithms.newton_no_ls import NewtonNoLS
from modopt.core.optimization_algorithms.quasi_newton_no_ls import QuasiNewtonNoLS

performance = {}
history = {}
time_loop = 20
for i, prob in enumerate([prob1, prob2, prob3]):
    print('\nProblem:', prob.problem_name)
    print('='*50)
    
    for Solver in [SteepestDescentNoLS, NewtonNoLS, QuasiNewtonNoLS, SteepestDescent, Newton, QuasiNewton]:
        if Solver.__name__.endswith('NoLS'):
            alg = Solver.__name__[:-4]
        else:
            alg = Solver.__name__ + '-LS'
    
        print(f'\t{Solver.__name__} \n\t------------------------')
        start_time = time.time()
        for i in range(time_loop):
            outs = ['opt', 'obj'] if i == (time_loop-1) else []
            turn_off = False if i == (time_loop-1) else True
            optimizer = Solver(prob, **{'maxiter': 200, 'opt_tol': 1e-6}, readable_outputs=outs, turn_off_outputs=turn_off)
            results   = optimizer.solve()
        opt_time = (time.time() - start_time) / time_loop
        success = results['converged']

        niter    = results['niter']
        nev      = results['total_callbacks']
        o_evals  = results['obj_evals']
        g_evals  = results['grad_evals']
        h_evals  = results['hess_evals']

        objective  = results['objective']
        optimality = results['optimality']
        x_opt      = results['x']

        print('\tTime:',            opt_time)
        print('\tSuccess:',         success)
        print('\tIterations:',      niter)
        print('\tEvaluations:',     nev)
        print('\tObj evals:',       o_evals)
        print('\tGrad evals:',      g_evals)
        print('\tHess evals:',      h_evals)
        print('\tOptimized vars:',  x_opt)
        print('\tOptimized obj:',   objective)
        print('\tOptimality:',      optimality)

        history[prob.problem_name, alg] = {'obj_hist': np.loadtxt(f"{results['out_dir']}/obj.out"),
                                           'opt_hist': np.loadtxt(f"{results['out_dir']}/opt.out")}

        performance[prob.problem_name, alg] = {'time': opt_time,
                                               'success': success,
                                               'niter': niter,
                                               'nev': nev,
                                               'objective': objective,
                                               'optimality': optimality}

    plt.figure()
    for alg in ['Newton', 'QuasiNewton', 'SteepestDescent-LS', 'Newton-LS', 'QuasiNewton-LS']:
        plt.semilogy(history[prob.problem_name, alg]['obj_hist'], label=alg)
    plt.xlabel('Iterations')
    plt.ylabel('Objective')
    plt.title(f'{prob.problem_name} minimization')
    plt.legend()
    plt.grid()
    plt.savefig(f"{prob.problem_name}-objective.pdf")
    plt.close()

    plt.figure()
    for alg in ['Newton', 'QuasiNewton', 'SteepestDescent-LS', 'Newton-LS', 'QuasiNewton-LS']:
        plt.semilogy(history[prob.problem_name, alg]['opt_hist'], label=alg)
    plt.xlabel('Iterations')
    plt.ylabel('Optimality')
    plt.title(f'{prob.problem_name} minimization')
    plt.legend()
    plt.grid()
    plt.savefig(f"{prob.problem_name}-optimality.pdf")
    plt.close()

# Print performance
print('\nPerformance')
print('='*50)
for key, value in performance.items():
    print(f"{str(key):45}:", value)

from modopt.benchmarking import plot_performance_profiles
plot_performance_profiles(performance, save_figname='performance.pdf')