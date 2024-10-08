'''Scaling study on optimizers and AMLs (Cantilever)'''

import numpy as np
import matplotlib.pyplot as plt

from modopt import PySLSQP, SNOPT, IPOPT, COBYQA, TrustConstr
from modopt.postprocessing import load_variables
from modopt.utils.profiling import profiler, time_profiler


from examples.ex_15_1cantilever_beam_fd import get_problem as get_fd_prob
from examples.ex_15_2cantilever_beam_casadi import get_problem as get_ca_prob
from examples.ex_15_3cantilever_beam_csdl import get_problem as get_csdl_prob
from examples.ex_15_4cantilever_beam_jax import get_problem as get_jax_prob
from examples.ex_15_5cantilever_beam_openmdao import get_problem as get_om_prob

num_elements = [20, 50, 100, 200, 300, 500]
sols = [23927.15, 23762.15, 23724.725, 23713.58, 23712.21, 23711.245]
obj_error = 12 # 0.05 percent of 24000

get_probs = [get_fd_prob, get_ca_prob, get_csdl_prob, get_jax_prob, get_om_prob]
methods   = ['FD', 'CasADi', 'CSDL', 'Jax', 'OpenMDAO']

history      = {}
performance  = {}

def print_stats_and_save_performance(prob, solver, results, x, f, i, 
                                     c_time, o_time, perf_dict, hist_dict):
    nev       = prob._callback_count
    o_evals   = prob._obj_count
    g_evals   = prob._grad_count
    h_evals   = prob._hess_count

    exec_time = o_time + c_time

    bd_viol_l  = np.maximum(0, prob.x_lower - x)
    bd_viol_u  = np.maximum(0, x - prob.x_upper)

    con         = prob._compute_constraints(x)
    con_viol_l  = np.maximum(0, prob.c_lower - con)
    con_viol_u  = np.maximum(0, con - prob.c_upper)

    feas = np.sum(bd_viol_l) + np.sum(bd_viol_u) + np.sum(con_viol_l) + np.sum(con_viol_u)

    success = np.isclose(f, sols[i], atol=obj_error) and feas < 1e-6

    print('\tTime:', exec_time, 's',    f'(compile [{c_time}] + optimize[{o_time}])')
    print('\tSuccess:', success)
    print('\tEvaluations:', nev)
    print('\tObj evals:', o_evals)
    print('\tGrad evals:', g_evals)
    print('\tHess evals:', h_evals)
    print('\tOptimized vars:', x)
    print('\tOptimized obj:', f)
    
    print('\tFeasibility:', feas)
    perf_dict[prob.problem_name, solver] = {'time': exec_time,
                                            'c_time': c_time,
                                            'o_time': o_time,
                                            'success': success,
                                            'nev': nev,
                                            'nfev': o_evals,
                                            'ngev': g_evals,
                                            'nhev': h_evals,
                                            'objective': f,
                                            'feasibility': feas}

    obj_hist = load_variables(f"{results['out_dir']}/record.hdf5", 'obj')['callback_obj']
    hist_dict[prob.problem_name, solver] = obj_hist

if __name__ == '__main__':

    for i, n_el in enumerate(num_elements):
        for get_prob, method in zip(get_probs, methods):

            if method == 'FD' and n_el >= 100:
                continue

            _compile_prob = time_profiler()(lambda n: get_prob(n))
            prob, compile_time = _compile_prob(n_el)

            print('\nProblem:', prob.problem_name)
            print('='*50)

            # # COBYQA
            # alg = 'COBYQA'
            # print(f'\t{alg} \n\t------------------------')
            # optimizer = COBYQA(prob, solver_options={'maxfev': 1000}, recording=True)
            
            # _solve = time_profiler()(lambda: optimizer.solve())
            # results, opt_time = _solve()
            
            # success  = optimizer.results['success']
            # print_stats_and_save_performance(prob, alg, results, results['x'], results['fun'], success,
            #                                  compile_time, opt_time, performance, history)


            # PySLSQP
            alg = 'PySLSQP'
            if n_el >= 200:
                performance[prob.problem_name, alg] = {'time': 1e6,
                                                       'c_time': compile_time,
                                                       'o_time': 1e6,
                                                       'success': False,
                                                       'nev': 1e6,
                                                       'nfev': 1e6,
                                                       'ngev': 1e6,
                                                       'nhev': 1e6,
                                                       'objective': 1e6,
                                                       'feasibility': 1e6}
            else:
                print(f'\t{alg} \n\t------------------------')
                optimizer = PySLSQP(prob, solver_options={'maxiter': 1000, 'acc': 1e-6, 'iprint': 0}, 
                                    recording=True)
                
                _solve = time_profiler()(lambda: optimizer.solve())
                results, opt_time = _solve()

                # success = results['success']
                print_stats_and_save_performance(prob, alg, results, results['x'], results['objective'], i, 
                                                compile_time, opt_time, performance, history)

            # SNOPT
            alg = 'SNOPT'
            print(f'\t{alg} \n\t------------------------')
            optimizer = SNOPT(prob, solver_options={'Major iterations': 1000, 'Major optimality': 1e-7, 'Verbose': False},
                              recording=True)

            _solve = time_profiler()(lambda: optimizer.solve())
            results, opt_time = _solve()
            
            # success = results['info']==1
            print_stats_and_save_performance(prob, alg, results, results['x'], results['objective'], i, 
                                             compile_time, opt_time, performance, history)

            sn_sol = results['x'] * 1.0

            # TrustConstr
            alg = 'TrustConstr'
            print(f'\t{alg} \n\t------------------------')
            optimizer = TrustConstr(prob, solver_options={'maxiter': 300, 'gtol':1e-2, 'xtol':1e-6, 'ignore_exact_hessian':True}, 
                                    recording=True)

            _solve = time_profiler()(lambda: optimizer.solve())
            results, opt_time = _solve()
            
            # success = results['success']
            print_stats_and_save_performance(prob, alg, results, results['x'], results['obj'], i,
                                             compile_time, opt_time, performance, history)

            tc_sol = results['x'] * 1.0

            # IPOPT
            alg = 'IPOPT'
            print(f'\t{alg} \n\t------------------------')
            optimizer = IPOPT(prob, solver_options={'max_iter': 200, 'tol': 1e-3, 'print_level': 0,'accept_after_max_steps': 10},
                            recording=True)

            _solve = time_profiler()(lambda: optimizer.solve())
            results, opt_time = _solve()
            
            print_stats_and_save_performance(prob, alg, results, results['x'], results['f'], i,
                                             compile_time, opt_time, performance, history)

            if (method in ['CasADi', 'Jax'] and n_el <= 500):
                _compile_prob = time_profiler()(lambda n: get_prob(n, order=2))
                prob, compile_time = _compile_prob(n_el)

                print('\nProblem:', prob.problem_name+'_ord2')
                print('='*50)

                # TrustConstr-2
                alg = 'TrustConstr-2'
                print(f'\t{alg} \n\t------------------------')
                optimizer = TrustConstr(prob, solver_options={'maxiter': 100, 'gtol':1e-3, 'xtol':1e-6}, 
                                        recording=True)

                _solve = time_profiler()(lambda: optimizer.solve())
                results, opt_time = _solve()
                
                # success = results['success']
                print_stats_and_save_performance(prob, alg, results, results['x'], results['obj'], i,
                                                 compile_time, opt_time, performance, history)

                tc_sol = results['x']

                # IPOPT-2
                alg = 'IPOPT-2'
                print(f'\t{alg} \n\t------------------------')
                optimizer = IPOPT(prob, solver_options={'max_iter': 100, 'tol': 1e-3, 'print_level': 0, 'accept_after_max_steps': 10, 'hessian_approximation': 'exact'},
                                recording=True)

                _solve = time_profiler()(lambda: optimizer.solve())
                results, opt_time = _solve()
                
                print_stats_and_save_performance(prob, alg, results, results['x'], results['f'], i,
                                                 compile_time, opt_time, performance, history)

            algs = ['PySLSQP', 'SNOPT', 'TrustConstr', 'IPOPT']
            if n_el >= 200:
                algs = ['SNOPT', 'TrustConstr', 'IPOPT']
            if method in ['CasADi', 'Jax']:
                algs += ['TrustConstr-2', 'IPOPT-2']
            

            plt.figure()
            for alg in algs:
                y_data = history[prob.problem_name, alg]
                plt.semilogy(y_data, label=f"{alg} ({len(y_data)})")
            plt.xlabel('Evaluations')
            plt.ylabel('Objective')
            plt.title(f'{prob.problem_name} minimization')
            plt.legend()
            plt.grid()
            plt.savefig(f"{prob.problem_name}-objective-cb.pdf")
            plt.close()

    # Plot time scaling - Optimizers (for Jax)
    plt.figure()
    for alg in ['PySLSQP', 'SNOPT', 'TrustConstr', 'IPOPT', 'TrustConstr-2', 'IPOPT-2']:
        x_data = [n_el for n_el in num_elements if performance[f"cantilever_{n_el}_jax", alg]['success']]
        y_data = [performance[f"cantilever_{n_el}_jax", alg]['time'] for n_el in num_elements if performance[f"cantilever_{n_el}_jax", alg]['success']]
        plt.semilogy(x_data, y_data, label=f"{alg}")

    plt.xlabel('Number of elements')
    plt.ylabel('Optimization time [s]')
    plt.title('Time scaling of optimization algorithms [Jax]')
    plt.legend()
    plt.grid()
    plt.savefig(f'cantilever_optimizers_time_scaling_semilogy.pdf')
    plt.close()

    # Plot time scaling - AMLs (for IPOPT/-2)
    plt.figure()
    for aml in ['fd', 'casadi', 'csdl', 'jax', 'om']:
        x_data = num_elements
        if aml == 'fd':
            x_data = [20, 50]
        y_data = [performance[f"cantilever_{n_el}_{aml}", 'IPOPT']['time'] for n_el in x_data]
        plt.semilogy(x_data, y_data, label=f"{aml}")

    for aml in ['casadi', 'jax']:
        x_data = num_elements
        y_data = [performance[f"cantilever_{n_el}_{aml}", 'IPOPT-2']['time'] for n_el in x_data]
        plt.semilogy(x_data, y_data, label=f"{aml}-2")

    plt.xlabel('Number of elements')
    plt.ylabel('Optimization time [s]')
    plt.title('Time scaling of AMLs [IPOPT]')
    plt.legend()
    plt.grid()
    plt.savefig(f'cantilever_aml_time_scaling_semilogy.pdf')
    plt.close()

    # Print performance
    print('\nPerformance')
    print('='*50)
    for key, value in performance.items():
        print(f"{str(key):40}:", value)

    # Performance without IPOPT-2 and TrustConstr-2
    new_performance = {}
    for key, value in performance.items():
        if key[1] in ['TrustConstr-2', 'IPOPT-2']:
            continue
        new_performance[key] = value

    from modopt.benchmarking import plot_performance_profiles
    plot_performance_profiles(new_performance, save_figname='performance.pdf')

    import pickle

    # Writing the dictionary to a file
    with open('obj_history.pkl', 'wb') as file:
        pickle.dump(history, file)

    # Reading the dictionary from the file
    with open('obj_history.pkl', 'rb') as file:
        loaded_history = pickle.load(file)

    print(loaded_history["cantilever_50_jax", 'IPOPT-2'])

    with open('performance.pkl', 'wb') as file:
        pickle.dump(performance, file)

    with open('performance.pkl', 'rb') as file:
        loaded_performance = pickle.load(file)

    print(loaded_performance["cantilever_50_jax", 'IPOPT-2'])


    with open('new_performance.pkl', 'wb') as file:
        pickle.dump(new_performance, file)

    with open('new_performance.pkl', 'rb') as file:
        loaded_new_performance = pickle.load(file)

    print(loaded_new_performance["cantilever_50_jax", 'IPOPT'])