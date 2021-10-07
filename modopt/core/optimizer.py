import numpy as np
import pandas
import time

from modopt.utils.options_dictionary import OptionsDictionary
from modopt.utils.general_utils import pad_name
from io import StringIO


class Optimizer(object):
    def __init__(self, problem, **kwargs):
        self.options = OptionsDictionary()
        problem._setup()
        self.prob_options = problem.options
        self.problem_name = problem.problem_name

        # try methods required for a specific optimizer are available inside the Problem subclass (inspect package)

        self.problem = problem
        self.options.declare('outputs', types=dict)
        self.options.declare('max_itr', default=1000, types=int)
        self.options.declare('formulation', default='rs', types=str)
        self.options.declare('opt_tol', default=1e-8, types=float)
        self.options.declare('feas_tol', default=1e-8, types=float)

        self.initialize()
        self.options.update(kwargs)

        self.outputs = self.options['outputs']
        for key in self.outputs:
            self.outputs[key] = np.array([])
        # for key, value in kwargs.items():
        #     self.outputs[key] = np.array([])
        # if value == 2:
        #     self.outputs[key] = np.array([])

        # Initialize a new file for saving xk eg., from callback() save_xk()
        name = self.problem_name
        nx = self.problem.nx
        x = self.problem.x.get_data()
        with open(name + '_x.out', 'w') as f:
            np.savetxt(f, x.reshape(1, nx))

        self.setup()

    def setup(self):
        pass

    # # only supports arrays
    # # key is the name of the output, and
    # # value is the dimension (1 or 2 ?) of the array input in each iteration
    # def declare_outputs(self, **kwargs):
    #     self.outputs = {}
    #     for key, value in kwargs.items():
    #         self.outputs[key] = np.array([])
    #         # if value == 2:
    #         #     self.outputs[key] = np.array([])

    def update_outputs(self, **kwargs):
        name = self.problem_name

        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):

                with open(name + '_' + key + '.out', 'a') as f:
                    np.savetxt(f, value.reshape(1, len(value)))
                self.outputs[key] = np.append(
                    self.outputs[key], value.reshape(1, len(value)))
            else:
                self.outputs[key] = np.append(self.outputs[key], value)
                '''save to table'''

        # Saving new x iterate on file

        # name = self.problem_name
        # for key, value in kwargs:

        # x = kwargs['x']
        # nx = self.problem.nx
        # if 'x' in kwargs:
        #     with open(name + '_x.out', 'a') as f:
        #         np.savetxt(f, x.reshape(1, nx))

        # # Saving new constraints on file
        # if 'con' in kwargs:
        #     c = kwargs['con']
        #     nc = self.problem.nc
        #     if 'con' in kwargs:
        #         with open(name + '_con.out', 'a') as f:
        #             np.savetxt(f, c.reshape(1, nc))

        #     # Saving new Lagrange multipliers on file
        #     lag = kwargs['lag_mult']
        #     if 'lag_mult' in kwargs:
        #         with open(name + '_lag_mult.out', 'a') as f:
        #             np.savetxt(f, lag.reshape(1, nc))

        # # Saving optimization progress on file (Modify to append one row at the end. Instead of storing all the arrays, we only store the current iteration and save everything else on file)
        # pandas.set_option('display.float_format', '{:.2E}'.format)
        # table = pandas.DataFrame({
        #     "Major": kwargs['itr'],
        #     "Obj": kwargs['obj'],
        #     "Opt": kwargs['opt'],
        #     "Time": kwargs['time']
        # })

        # if 'num_f_evals' in kwargs:
        #     table['f_evals'] = kwargs['num_f_evals']
        # if 'num_g_evals' in kwargs:
        #     table['g_evals'] = kwargs['num_g_evals']
        # if 'step' in kwargs:
        #     table['Step'] = kwargs['step']
        # if 'feas' in kwargs:
        #     table['Feas'] = kwargs['feas']
        # if 'penalty' in kwargs:
        #     table['Penalty'] = kwargs['penalty']
        # if 'merit' in kwargs:
        #     table['Merit'] = kwargs['merit']

        # with open(name + '_print.out', 'w') as f:
        #     f.writelines(table.to_string(index=False))

    def update_outputs_dict(self, **kwargs):
        self.outputs['itr_array'] = kwargs['itr']
        self.outputs['x_array'] = kwargs['x']
        self.outputs['obj_array'] = kwargs['obj']
        self.outputs['opt_array'] = kwargs['opt']
        self.outputs['time_array'] = kwargs['time']

        if 'con' in kwargs:
            self.outputs['lag_mult_array'] = kwargs['con']
        if 'lag_mult' in kwargs:
            self.outputs['con_array'] = kwargs['lag_mult']
        if 'num_f_evals' in kwargs:
            self.outputs['num_f_evals_array'] = kwargs['num_f_evals']
        if 'num_g_evals' in kwargs:
            self.outputs['num_g_evals_array'] = kwargs['num_g_evals']
        if 'step' in kwargs:
            self.outputs['step_array'] = kwargs['step']
        if 'feas' in kwargs:
            self.outputs['feas_array'] = kwargs['feas']
        if 'penalty' in kwargs:
            self.outputs['penalty_array'] = kwargs['penalty']
        if 'merit' in kwargs:
            self.outputs['merit_array'] = kwargs['merit']

    def print_results(self, **kwargs):
        # Testing to verify the design variable data
        # print(np.loadtxt(self.problem_name+'_x.out') - self.outputs['x_array'])
        print("\n", "\t" * 1, "modOpt summary:")
        print("\t" * 1, "===============", "\n")
        print("\t" * 1, "Problem", "\t" * 3, ':', self.problem_name)
        print("\t" * 1, "Solver", "\t" * 3, ':', self.solver_name)
        print("\t" * 1, "Objective", "\t" * 3, ':',
              self.outputs['obj_array'][-1])
        print("\t" * 1, "Optimality", "\t" * 3, ':',
              self.outputs['opt_array'][-1])

        if self.outputs['feas_array'] is not None:
            print("\t" * 1, "Feasibility", "\t" * 2, ':',
                  self.outputs['feas_array'][-1])

        print("\t" * 1, "Total time:", "\t" * 3, ':', self.total_time)
        print("\t" * 1, "Major iterations", "\t" * 2, ':',
              self.outputs['itr_array'][-1])

        if self.outputs['num_f_evals_array'] is not None:
            print("\t" * 1, "Total function evaluations", "\t" * 1, ':',
                  self.outputs['num_f_evals_array'][-1])
        if self.outputs['num_g_evals_array'] is not None:
            print("\t" * 1, "Total gradient evaluations", "\t" * 1, ':',
                  self.outputs['num_g_evals_array'][-1])

        allowed_keys = {
            'optimal_variables', 'optimal_constraints',
            'optimal_lag_mult', 'summary_table', 'compact_print'
        }
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, val) for key, val in kwargs.items()
                             if key in allowed_keys)

        if self.optimal_variables:
            print("\t" * 1, "Optimal variables", "\t" * 2, ':',
                  self.outputs['x_array'][-1])

        if self.optimal_constraints:
            print("\t" * 1, "Optimal constraints", "\t" * 2, ':',
                  self.outputs['con_array'][-1])

        if self.optimal_lag_mult:
            print("\t" * 1, "Optimal Lagrange multipliers", "\t" * 1,
                  ':', self.outputs['lag_mult_array'][-1])

        print("\n", "\t", "===== End of summary =====", "\n")

        # Print optimization summary table
        if self.summary_table:

            print("\n", "\t" * 2, "modOpt summary table:")
            print("\t" * 2, "=====================", "\n")
            name = self.problem_name
            with open(name + '_print.out', 'r') as f:
                # lines = f.readlines()
                lines = f.read().splitlines()

            # Print all iterations
            if not (self.compact_print):
                for i in range(len(lines)):
                    print(lines[i])

            # Print only itrs_to_print above the itr_threshold
            else:
                itr_threshold = 10
                itrs_to_print = 5

                if np.size(self.outputs['itr_array']) <= itr_threshold:
                    for i in range(len(lines)):
                        print(lines[i])

                else:
                    idx = np.linspace(
                        0,
                        np.size(self.outputs['itr_array']) - 1,
                        itrs_to_print,
                        dtype='int')

                    # Account for the column label index
                    idx = np.append(0, idx + 1)
                    for i in idx:
                        print(lines[i])

    def check_first_derivatives(self, x):
        obj = self.obj
        grad = self.grad

        nx = self.problem.nx
        nc = self.problem.nc
        constrained = False
        if nc != 0:
            constrained = True
            con = self.con
            jac = self.jac

        h = 1e-9

        grad_fd = np.full((nx, ), obj(x), dtype=float)
        if constrained:
            jac_fd = np.outer(con(x), np.ones((nx, ), dtype=float))

        for i in range(nx):
            e = np.zeros((nx, ), dtype=float)
            e[i] = h

            grad_fd[i] -= obj(x + e)

            if constrained:
                jac_fd[:, i] -= con(x + e)

        grad_fd /= -h
        grad_exact = grad(x)

        if constrained:
            jac_fd /= -h
            jac_exact = jac(x)

        EPSILON = 1e-10

        # print('grad_exact:', grad_exact)
        # print('grad_fd:', grad_fd)
        # print('jac_exact:', jac_exact)
        # print('jac_fd:', jac_fd)

        grad_abs_error = np.absolute(grad_fd - grad_exact)
        grad_rel_error = grad_abs_error / (
            np.absolute(grad_fd) + EPSILON
        )  # fd is assumed to give the actual gradient

        if constrained:
            jac_abs_error = np.absolute(jac_fd - jac_exact)
            jac_rel_error = jac_abs_error / np.linalg.norm(
                jac_fd, 'fro')
            # jac_rel_error = jac_abs_error / (np.absolute(jac_fd) + EPSILON)
            # jac_rel_error = jac_abs_error / (np.absolute(jac_exact.toarray()) + EPSILON)

        out_buffer = StringIO()

        header = "{0} | {1} | {2} | {3} | {4} "\
                            .format(
                                pad_name('Derivative type', 8, quotes=False),
                                pad_name('Calc norm', 10),
                                pad_name('FD norm', 10),
                                pad_name('Abs error norm', 10),
                                pad_name('Rel error norm', 10),
                            )

        out_buffer.write('\n' + header + '\n')
        out_buffer.write('-' * len(header) + '\n' + '\n')

        deriv_line = "{0} | {1:.4e} | {2:.4e} | {3:.4e}     | {4:.4e}    "
        grad_line = deriv_line.format(
            pad_name('Gradient', 15, quotes=False),
            np.linalg.norm(grad_exact),
            np.linalg.norm(grad_fd),
            np.linalg.norm(grad_abs_error),
            np.linalg.norm(grad_rel_error),
        )

        out_buffer.write(grad_line + '\n')
        if constrained:
            jac_line = deriv_line.format(
                pad_name('Jacobian', 15, quotes=False),
                np.linalg.norm(jac_exact),
                np.linalg.norm(jac_fd),
                np.linalg.norm(jac_abs_error),
                np.linalg.norm(jac_rel_error),
            )

            out_buffer.write(jac_line + '\n')

        print(out_buffer.getvalue())