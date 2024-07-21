'''Benchmark optimization algorithms on four simple problems'''

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
sol3  = np.array([0.5, 0.5])

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
    
    # Steepest-descent without line search
    print('\tSteepestDescentNoLS \n\t-----')
    start_time = time.time()
    for i in range(time_loop):
        outs = ['opt', 'obj'] if i == (time_loop-1) else []
        turn_off = False if i == (time_loop-1) else True
        optimizer = SteepestDescentNoLS(prob, **{'maxiter': 200, 'opt_tol': 1e-6}, readable_outputs=outs, turn_off_outputs=turn_off)
        optimizer.solve()
    opt_time = (time.time() - start_time) / time_loop
    success = optimizer.results['converged']
    print('\tTime:', opt_time)
    print('\tSuccess:', success)
    print('\tIterations:', optimizer.results['niter'])
    print('\tEvaluations', optimizer.results['niter']*2 + 2)
    print('\tOptimized vars:', optimizer.results['x'])
    print('\tOptimized obj:', optimizer.results['objective'])
    print('\tOptimality:', optimizer.results['optimality'])

    history[prob.problem_name, 'SteepestDescent'] = {'obj_hist': np.loadtxt(f"{optimizer.out_dir}/obj.out"),
                                                         'opt_hist': np.loadtxt(f"{optimizer.out_dir}/opt.out")}

    performance[prob.problem_name, 'SteepestDescent'] = {'time': opt_time,
                                                        'success': success,
                                                        'niter': optimizer.results['niter'],
                                                        'nev': optimizer.results['niter']*2,
                                                        'objective': optimizer.results['objective'],
                                                        'optimality': optimizer.results['optimality']}

    # Newton without line search
    print('\tNewtontNoLS \n\t-----')
    start_time = time.time()
    for i in range(time_loop):
        outs = ['opt', 'obj'] if i == (time_loop-1) else []
        turn_off = False if i == (time_loop-1) else True
        optimizer = NewtonNoLS(prob, **{'maxiter': 200, 'opt_tol': 1e-6}, readable_outputs=outs, turn_off_outputs=turn_off)
        optimizer.solve()
    opt_time = (time.time() - start_time) / time_loop
    success = optimizer.results['converged']
    print('\tTime:', opt_time)
    print('\tSuccess:', success)
    print('\tIterations:', optimizer.results['niter'])
    print('\tEvaluations', optimizer.results['niter']*3 + 3)
    print('\tOptimized vars:', optimizer.results['x'])
    print('\tOptimized obj:', optimizer.results['objective'])
    print('\tOptimality:', optimizer.results['optimality'])

    history[prob.problem_name, 'Newton'] = {'obj_hist': np.loadtxt(f"{optimizer.out_dir}/obj.out"),
                                                'opt_hist': np.loadtxt(f"{optimizer.out_dir}/opt.out")}

    performance[prob.problem_name, 'Newton'] = {'time': opt_time,
                                                'success': success,
                                                'niter': optimizer.results['niter'],
                                                'nev': optimizer.results['niter']*3,
                                                'objective': optimizer.results['objective'],
                                                'optimality': optimizer.results['optimality']}

    # QuasiNewton without line search
    print('\tQuasiNewtonNoLS \n\t-----')
    start_time = time.time()
    for i in range(time_loop):
        outs = ['opt', 'obj'] if i == (time_loop-1) else []
        turn_off = False if i == (time_loop-1) else True
        optimizer = QuasiNewtonNoLS(prob, **{'maxiter': 200, 'opt_tol': 1e-6}, readable_outputs=outs, turn_off_outputs=turn_off)
        optimizer.solve()
    opt_time = (time.time() - start_time) / time_loop
    success = optimizer.results['converged']
    print('\tTime:', opt_time)
    print('\tSuccess:', success)
    print('\tIterations:', optimizer.results['niter'])
    print('\tEvaluations', optimizer.results['niter']*2 + 2)
    print('\tOptimized vars:', optimizer.results['x'])
    print('\tOptimized obj:', optimizer.results['objective'])
    print('\tOptimality:', optimizer.results['optimality'])

    history[prob.problem_name, 'QuasiNewton'] = {'obj_hist': np.loadtxt(f"{optimizer.out_dir}/obj.out"),
                                                     'opt_hist': np.loadtxt(f"{optimizer.out_dir}/opt.out")}

    performance[prob.problem_name, 'QuasiNewton'] = {'time': opt_time,
                                                    'success': success,
                                                    'niter': optimizer.results['niter'],
                                                    'nev': optimizer.results['niter']*2,
                                                    'objective': optimizer.results['objective'],
                                                    'optimality': optimizer.results['optimality']}
    
    # Steepest-descent with line search
    print('\tSteepestDescent \n\t-----')
    start_time = time.time()
    for i in range(time_loop):
        outs = ['opt', 'obj'] if i == (time_loop-1) else []
        turn_off = False if i == (time_loop-1) else True
        optimizer = SteepestDescent(prob, **{'maxiter': 200, 'opt_tol': 1e-6}, readable_outputs=outs, turn_off_outputs=turn_off)
        optimizer.solve()
    opt_time = (time.time() - start_time) / time_loop
    success = optimizer.results['converged']
    print('\tTime:', opt_time)
    print('\tSuccess:', success)
    print('\tIterations:', optimizer.results['niter'])
    print('\tEvaluations', sum(optimizer.results[key] for key in ['nfev', 'ngev']))
    print('\tOptimized vars:', optimizer.results['x'])
    print('\tOptimized obj:', optimizer.results['objective'])
    print('\tOptimality:', optimizer.results['optimality'])

    history[prob.problem_name, 'SteepestDescent-LS'] = {'obj_hist': np.loadtxt(f"{optimizer.out_dir}/obj.out"),
                                                     'opt_hist': np.loadtxt(f"{optimizer.out_dir}/opt.out")}

    performance[prob.problem_name, 'SteepestDescent-LS'] = {'time': opt_time,
                                                         'success': success,
                                                         'niter': optimizer.results['niter'],
                                                         'nev': sum(optimizer.results[key] for key in ['nfev', 'ngev']),
                                                         'objective': optimizer.results['objective'],
                                                         'optimality': optimizer.results['optimality']}
    
    # Newton with line search
    print('\tNewton \n\t-----')
    start_time = time.time()
    for i in range(time_loop):
        outs = ['opt', 'obj'] if i == (time_loop-1) else []
        turn_off = False if i == (time_loop-1) else True
        optimizer = Newton(prob, **{'maxiter': 200, 'opt_tol': 1e-6}, readable_outputs=outs, turn_off_outputs=turn_off)
        optimizer.solve()
    opt_time = (time.time() - start_time) / time_loop
    success = optimizer.results['converged']
    print('\tTime:', opt_time)
    print('\tSuccess:', success)
    print('\tIterations:', optimizer.results['niter'])
    print('\tEvaluations', sum(optimizer.results[key] for key in ['nfev', 'ngev', 'nhev']))
    print('\tOptimized vars:', optimizer.results['x'])
    print('\tOptimized obj:', optimizer.results['objective'])
    print('\tOptimality:', optimizer.results['optimality'])

    history[prob.problem_name, 'Newton-LS'] = {'obj_hist': np.loadtxt(f"{optimizer.out_dir}/obj.out"),
                                               'opt_hist': np.loadtxt(f"{optimizer.out_dir}/opt.out")}

    performance[prob.problem_name, 'Newton-LS'] = {'time': opt_time,
                                                'success': success,
                                                'niter': optimizer.results['niter'],
                                                'nev': sum(optimizer.results[key] for key in ['nfev', 'ngev', 'nhev']),
                                                'objective': optimizer.results['objective'],
                                                'optimality': optimizer.results['optimality']}
    
    # QuasiNewton with line search
    print('\tQuasiNewton \n\t-----')
    start_time = time.time()
    for i in range(time_loop):
        outs = ['opt', 'obj'] if i == (time_loop-1) else []
        turn_off = False if i == (time_loop-1) else True
        optimizer = QuasiNewton(prob, **{'maxiter': 200, 'opt_tol': 1e-6}, readable_outputs=outs, turn_off_outputs=turn_off)
        optimizer.solve()
    opt_time = (time.time() - start_time) / time_loop
    success = optimizer.results['converged']
    print('\tTime:', opt_time)
    print('\tSuccess:', success)
    print('\tIterations:', optimizer.results['niter'])
    print('\tEvaluations', sum(optimizer.results[key] for key in ['nfev', 'ngev']))
    print('\tOptimized vars:', optimizer.results['x'])
    print('\tOptimized obj:', optimizer.results['objective'])
    print('\tOptimality:', optimizer.results['optimality'])

    history[prob.problem_name, 'QuasiNewton-LS'] = {'obj_hist': np.loadtxt(f"{optimizer.out_dir}/obj.out"),
                                                 'opt_hist': np.loadtxt(f"{optimizer.out_dir}/opt.out")}

    performance[prob.problem_name, 'QuasiNewton-LS'] = {'time': opt_time,
                                                        'success': success,
                                                        'niter': optimizer.results['niter'],
                                                        'nev': sum(optimizer.results[key] for key in ['nfev', 'ngev']),
                                                        'objective': optimizer.results['objective'],
                                                        'optimality': optimizer.results['optimality']}

    plt.figure()
    for optimizer in ['Newton', 'QuasiNewton', 'SteepestDescent-LS', 'Newton-LS', 'QuasiNewton-LS']:
        plt.semilogy(history[prob.problem_name, optimizer]['obj_hist'], label=optimizer)
    plt.xlabel('Iterations')
    plt.ylabel('Objective')
    plt.title(f'{prob.problem_name} minimization')
    plt.legend()
    plt.grid()
    plt.savefig(f"{prob.problem_name}-objective.pdf")
    plt.close()

    plt.figure()
    for optimizer in ['Newton', 'QuasiNewton', 'SteepestDescent-LS', 'Newton-LS', 'QuasiNewton-LS']:
        plt.semilogy(history[prob.problem_name, optimizer]['opt_hist'], label=optimizer)
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