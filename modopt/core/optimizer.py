import numpy as np
import scipy as sp
import pandas
import time

from modopt.utils.options_dictionary import OptionsDictionary
from modopt.utils.general_utils import pad_name
from io import StringIO


class Optimizer(object):
    def __init__(self, problem, **kwargs):

        self.options = OptionsDictionary()
        self.prob_options = problem.options
        self.problem_name = problem.problem_name

        # try methods required for a specific optimizer are available inside the Problem subclass (inspect package)

        self.problem = problem

        self.options.declare('max_itr', default=1000, types=int)
        self.options.declare('formulation', default='rs', types=str)
        self.options.declare('opt_tol', default=1e-8, types=float)
        self.options.declare('feas_tol', default=1e-8, types=float)

        self.initialize()
        self.options.update(kwargs)

        self._setup()

    def _setup(self):
        self.setup()
        name = self.problem_name
        self.outputs = {}
        fmt = self.default_outputs_format

        # Only user-specified outputs will be stored
        for key in self.options['outputs']:
            if isinstance(fmt[key], tuple):
                self.outputs[key] = np.empty((1, ) + fmt[key][1],
                                             dtype=fmt[key][0])

                with open(name + '_' + key + '.out', 'w') as f:
                    pass
            else:
                self.outputs[key] = np.array([], dtype=fmt[key])

    def setup():
        pass

    # def print_available_outputs(self, ): works only after initialization
    #     print(self.default_output_format)
    #  along with


# the outputs you wish to be stored in the outputs dictionary after each iteration

#         # Only user-specified outputs will be stored
#     for key in self.options['outputs']:
#         if len(fmt[key]) == 3:
#             self.outputs[key] = np.empty((1, ) + fmt[key][2],
#                                          dtype=fmt[key][1])

#             with open(name + '_' + key + '.out', 'w') as f:
#                 pass
#         else:
#             self.outputs[key] = np.array([], dtype=fmt[key[0]])

# def setup(self):
#     pass

# def print_available_outputs(self, ):
#     print(self.default_output_format[])

# Removes the first entry of the array when it was initialized as empty

    def run_post_processing(self):
        for key, value in self.outputs.items():
            if len(value.shape) >= 2:
                self.outputs[key] = self.outputs[key][1:]

    def update_outputs(self, **kwargs):
        name = self.problem_name
        pandas.set_option('display.float_format', '{:.2E}'.format)
        table = pandas.DataFrame({})

        for key, value in kwargs.items():
            # Only user-specified outputs will be stored
            # Multidimensional arrays will be flattened before writing to a file
            if key in self.outputs:
                if isinstance(value, np.ndarray):
                    with open(name + '_' + key + '.out', 'a') as f:
                        np.savetxt(f, value.reshape(1, value.size))
                    self.outputs[key] = np.append(
                        self.outputs[key],
                        value.reshape((1, ) + value.shape),
                        axis=0)
                else:
                    self.outputs[key] = np.append(self.outputs[key],
                                                  value)

                    table[key] = self.outputs[key]

                with open(name + '_print.out', 'w') as f:
                    f.writelines(table.to_string(index=False))

    def print_results(self, **kwargs):
        # Testing to verify the design variable data
        # print(np.loadtxt(self.problem_name+'_x.out') - self.outputs['x_array'])
        print("\n", "\t" * 1, "===============================")
        print("\t" * 1, "ModOpt final iteration summary:")
        print("\t" * 1, "===============================")

        max_string_length = 7
        for key in self.outputs:
            if len(key) > max_string_length:
                max_string_length = len(key)

        total_length = max_string_length + 5

        print("\t" * 1, "Problem", " " * (total_length - 7), ':',
              self.problem_name)
        print("\t" * 1, "Solver", " " * (total_length - 6), ':',
              self.solver_name)

        for key, value in self.outputs.items():
            if len(value.shape) == 1:
                print("\t" * 1, key, " " * (total_length - len(key)),
                      ':', value[-1])
            # if key == 'rho':
            #     print("\t" * 1, key, " " * (total_length - len(key)),
            #           ':', value[:, 0])

        allowed_keys = {'summary_table', 'compact_print'}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, val) for key, val in kwargs.items()
                             if key in allowed_keys)

        print("\t", "===============================")
        # print("\t", "End of final iteration summary")
        # print("\t", "==============================", "\n")

        # Print optimization summary table
        if self.summary_table:

            name = self.problem_name
            with open(name + '_print.out', 'r') as f:
                # lines = f.readlines()
                lines = f.read().splitlines()

            line_length = len(lines[0])

            print("\n")
            print("=" * max(line_length, 21), )
            print("modOpt summary table:".center(max(line_length, 21)))
            print("=" * max(line_length, 21), )

            # Number of iterations including zeroth iteration (after removing column label)
            num_itr = len(lines) - 1

            # Print all iterations
            if not (self.compact_print):
                for i in range(num_itr + 1):
                    print(lines[i])

            # Print only itrs_to_print above the itr_threshold
            else:
                itr_threshold = 10
                itrs_to_print = 5

                if len(lines) <= itr_threshold:
                    for i in range(num_itr):
                        print(lines[i])

                else:
                    idx = np.linspace(0,
                                      num_itr - 1,
                                      itrs_to_print,
                                      dtype='int')

                    # Account for the column label index
                    idx = np.append(0, idx + 1)
                    for i in idx:
                        print(lines[i])

            print("=" * max(line_length, 21), )

    def check_first_derivatives(self, x, method='rs'):
        obj = self.obj
        grad = self.grad

        if self.problem.ny == 0:
            nx = self.problem.nx
            nc = self.problem.nc
        else:
            nx = self.problem.n
            nc = self.problem.m

        constrained = False
        if nc != 0:
            constrained = True
            con = self.con
            jac = self.jac

        if method in ('cfs', 'surf'):
            print("INSIDE IF =================")
            y = self.problem.solve_residual_equations(
                x[:self.problem.nx])
            x[self.problem.nx:] = y

            self.problem.formulation = 'fs'

        grad_exact = grad(x)
        if constrained:
            jac_exact = jac(x)

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
        # grad_exact = grad(x)

        if constrained:
            jac_fd /= -h
            # jac_exact = jac(x)

        EPSILON = 1e-10

        # print('grad_exact:', grad_exact)
        # print('grad_fd:', grad_fd)
        # print('jac_exact:', jac_exact)
        # print('jac_fd:', jac_fd)

        grad_abs_error = np.absolute(grad_fd - grad_exact)
        # print('pf_px', grad_abs_error[:self.problem.nx])
        # print('pf_py', grad_abs_error[self.problem.nx:])
        grad_rel_error = grad_abs_error / (
            np.absolute(grad_fd) + EPSILON
        )  # fd is assumed to give the actual gradient

        if constrained:
            jac_abs_error = np.absolute(jac_fd - jac_exact)
            jac_rel_error = jac_abs_error / np.linalg.norm(
                jac_fd, 'fro')
            # print('pC_px', jac_abs_error[0, :self.problem.nx])
            # print(
            #     'pB+_px',
            #     np.linalg.norm(jac_abs_error[1:145, :self.problem.nx]))
            # print(
            #     'pB-_px',
            #     np.linalg.norm(
            #         jac_abs_error[145:289, :self.problem.nx]))
            # print(
            #     'pB+_py',
            #     np.linalg.norm(jac_abs_error[1:145, self.problem.nx:]))
            # print(
            #     'pB-_py',
            #     np.linalg.norm(jac_abs_error[145:289,
            #                                  self.problem.nx:]))

            # print(
            #     'pR+_px',
            #     np.linalg.norm(
            #         jac_abs_error[289:639, :self.problem.nx]))
            # print(jac_abs_error.shape)
            # print(
            #     'pR-_px',
            #     np.linalg.norm(
            #         jac_abs_error[639:989, :self.problem.nx]))
            # print(
            #     'pR+_py',
            #     np.linalg.norm(jac_abs_error[289:639,
            #                                  self.problem.nx:]))
            # print(
            #     'pR-_py',
            #     np.linalg.norm(jac_abs_error[639:989,
            #                                  self.problem.nx:]))

            # print('pR+_py_fd',
            #       np.linalg.norm(jac_fd[289:639, self.problem.nx:]))
            # print('pR-_py_fd',
            #       np.linalg.norm(jac_fd[639:989, self.problem.nx:]))

            # print(
            #     'pR+_py_exact',
            #     sp.sparse.linalg.norm(jac_exact[289:639,
            #                                     self.problem.nx:]))
            # print(
            #     'pR-_py_exact',
            #     sp.sparse.linalg.norm(jac_exact[639:989,
            #                                     self.problem.nx:]))

            # print('pC_py', jac_abs_error[0, self.problem.nx:])
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
                # sp.sparse.linalg.norm(jac_exact),
                np.linalg.norm(jac_fd),
                np.linalg.norm(jac_abs_error),
                np.linalg.norm(jac_rel_error),
            )

            out_buffer.write(jac_line + '\n')

        # print(out_buffer.getvalue())
