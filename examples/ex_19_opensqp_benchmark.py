'''Benchmark OpenSQP against other solvers on CUTEst problems (nx,nc<=100)'''

import numpy as np
from modopt import CUTEstProblem, optimize, OpenSQP
import pycutest
import time
import contextlib
import io
import gc # garbage collector

algs = ['SNOPT', 'IPOPT', 'PySLSQP', 'TrustConstr', 'OpenSQP']

performance = {}
history = {}
time_loop = 1
pc_prob = None
prob    = None

# Only import required problems based on the table
from modopt.benchmarking import filter_cutest_problems
prob_list = filter_cutest_problems(num_vars=[1,100], num_cons=[0,100])
print('# of problems after filtering:', len(prob_list))

# Remove some problems that cause import issues
remove_probs = ['DMN15102LS', 'DMN15103LS', 'DMN15332LS', 'DMN15333LS', 'DMN37142LS', 'DMN37143LS', 'BLEACHNG']
prob_list = [p for p in prob_list if p not in remove_probs]
print('# of problems after removing problems:', len(prob_list))

# SKIP problems for now that cause seg faults or 
# malloc errors with any of the solvers.
# Unfortunately, these problems need to be run separately 
# for solvers that have issues with them.
# This list might be different and needs to be updated 
# for different machines / OS / solver versions.
skip_probs = ['AVION2', # seg fault with IPOPT
              'BARDNE', # seg fault with OpenSQP
              'BOX3NE', # seg fault with PySLSQP
              'CHNRSBNE', # malloc error in Trust-Constr
              'MISRA1A', # malloc error in OpenSQP
              'MISRA1BLS', # seg fault
              'MISRA1C', # seg fault in OpenSQP
              'MISRA1CLS', # seg fault in OpenSQP
              'RAT42', # seg fault in OpenSQP
              'RAT43', # seg fault in PySLSQP
              ]
prob_list = [p for p in prob_list if p not in skip_probs]
print('# of problems after skipping problems:', len(prob_list))

for i, prob_name in enumerate(prob_list):
    # Import pycutest problem
    del pc_prob
    gc.collect()
    pc_prob = pycutest.import_problem(prob_name)
    print(f'[{i}.]', 'Problem name [num_vars, num_cons]:', prob_name, f'[{pc_prob.n}]', f'[{pc_prob.m}]')
    print('='*50)

    # Create modopt problem
    del prob
    gc.collect()
    prob = CUTEstProblem(cutest_problem=pc_prob)
    maxiter=250
    
    for alg in algs:
        solver = alg
        options = {}
        if alg=='IPOPT':
            options = {'hessian_approximation': 'limited-memory', 'limited_memory_max_history': 1000, 'max_iter':maxiter, 'print_level': 0, 'tol':1e-6}
        elif alg == 'TrustConstr':
            options = {'ignore_exact_hessian': True, 'maxiter': maxiter, 'gtol': 2e-5, 'xtol': 1e-100}
        elif alg == 'SNOPT':
            options = {'Verbose': False, 'Major optimality': 1.22e-4, 'Major iterations': maxiter, 'Major feasibility': 2e-6, "Verify level": -1, 
                       "Hessian frequency": 1000, "Hessian updates": 1000}
        elif alg == 'PySLSQP':
            options = {'maxiter': maxiter, 'iprint': 0, 'acc': 1e-6}
        elif alg == 'OpenSQP':
            options = {'maxiter': maxiter, 'opt_tol': 1.22e-4, 'feas_tol': 2e-6}

        print(f'\t{alg} \n\t------------------------')
        start_time = time.time()
        try:
            for i in range(time_loop-1):
                with contextlib.redirect_stdout(io.StringIO()):
                    if solver != 'OpenSQP':
                        results = optimize(prob, solver=solver, solver_options=options, recording=False, turn_off_outputs=True)
                    else:
                        results = OpenSQP(prob, maxiter=maxiter, opt_tol=options['opt_tol'], feas_tol=options['feas_tol'], recording=False, turn_off_outputs=True).solve()

            if solver != 'OpenSQP':
                results = optimize(prob, solver=solver, solver_options=options, recording=False)
            else:
                results = OpenSQP(prob, maxiter=maxiter, opt_tol=options['opt_tol'], feas_tol=options['feas_tol'], recording=False).solve()

            opt_time = (time.time() - start_time) / time_loop

            if alg == 'SNOPT':
                success = (results['info'] == 1)
            elif alg in ['IPOPT', 'IPOPT-2']:
                with open(f"{results['out_dir']}/ipopt_output.txt", 'r') as f:
                    ipopt_output = f.read()
                success = ('EXIT: Optimal Solution Found.' in ipopt_output)
            else:
                success  = results['success']
            
            nev         = prob._callback_count
            o_evals     = prob._obj_count
            g_evals     = prob._grad_count
            h_evals     = prob._hess_count
            objective   = prob._compute_objective(results['x'])

            x           = results['x']
            bd_viol_l   = np.maximum(0, prob.x_lower - x)
            bd_viol_u   = np.maximum(0, x - prob.x_upper)

            if prob.constrained:
                con         = prob._compute_constraints(x)
                con_viol_l  = np.maximum(0, prob.c_lower - con)
                con_viol_u  = np.maximum(0, con - prob.c_upper)
                feasibility = np.sum(bd_viol_l) + np.sum(bd_viol_u) + np.sum(con_viol_l) + np.sum(con_viol_u)
            else:
                feasibility = np.sum(bd_viol_l) + np.sum(bd_viol_u)

            if alg == 'OpenSQP':
                nev     = results['nfev'] + results['ngev']
                nev     = nev * 2 if prob.constrained else nev
                o_evals = results['nfev']
                g_evals = results['ngev']
                h_evals = 0
            elif alg == 'SNOPT':
                nev     = nev * 4 if prob.constrained else nev * 2
                
        except Exception as e:
            print(f'Error: {e}')
            success = False
            nev = 1e6
            o_evals = 1e6
            g_evals = 1e6
            h_evals = 1e6
            objective = 1e6
            opt_time = 1e6
            feasibility = 1e6

        performance[prob.problem_name, alg] = {'time': opt_time,
                                               'success': success,
                                               'nev': nev,
                                               'objective': objective,
                                               'feasibility': feasibility}

failed_probs = {alg: [] for alg in algs}
# Print performance
print('\nPerformance')
print('='*50)
for key, value in performance.items():
    print(f"{str(key):45}:", value)
    if value['success'] == False:
        failed_probs[key[1]].append(key[0])

from modopt.benchmarking import plot_performance_profiles
plot_performance_profiles(performance, save_figname='performance.pdf')