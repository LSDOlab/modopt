import numpy as np
import scipy as sp
import os

from modopt.utils.options_dictionary import OptionsDictionary
from modopt.utils.general_utils import pad_name
# from io import StringIO


class Optimizer(object):
    def __init__(self, problem, **kwargs):

        self.options = OptionsDictionary()
        self.problem = problem
        self.prob_options = problem.options
        self.problem_name = problem.problem_name
        # try methods required for a specific optimizer are available inside the Problem subclass (inspect package)

        self.solver_name = 'unnamed_solver'
        self.options.declare('formulation', default='rs', types=str)

        self.initialize()
        self.options.update(kwargs)
        self._setup()

    def _setup(self):
        # User defined optimizer-specific setup
        self.setup()

        name = self.problem_name
        self.outputs = {}
        outs = self.available_outputs
        dirName = name + '_outputs'
        
        # To print different outputs in different formats to different files
        if len (self.options['outputs']) > 0:
            # Create new directory for all the outputs of optimization
            try:
                os.mkdir(dirName)
            except FileExistsError:
                print("Directory ", dirName, " already exists")

            # Only user-specified outputs will be stored
            for key in self.options['outputs']:
                # TODO: Add a test for this
                if key not in outs:
                    raise ValueError(
                        'Declared unavailable output "{}"'.format(key))

                # Create new dictionaries for all user-specified outputs
                if isinstance(outs[key], tuple):
                    self.outputs[key] = np.empty((1, ) + outs[key][1],
                                                dtype=outs[key][0])

                    with open(dirName + '/' + key + '.out', 'w') as f:
                        pass
                else:
                    self.outputs[key] = np.array([], dtype=outs[key])
                    with open(dirName + '/' + key + '.out', 'w') as f:
                        pass
        
            # Write the header of the summary_table file
            self.scalar_keys = [out for out in outs.keys() if not isinstance(outs[out], tuple)]
            header =''
            for key in self.scalar_keys:
                if outs[key] in (int, np.int_, np.int32, np.int64):
                    header += "%16s " % key
                elif outs[key] in (float, np.float_, np.float32, np.float64):
                    header += "%16s " % key
            header += '\n'
            self.out_dict = {key: 0 for key in outs.keys()}

            with open(dirName + '/' + 'summary.out', 'w') as f:
                f.write(header)

    def setup(self, ):
        pass

    def run_post_processing(self):
        # Removes the first entry of the array when it was initialized as empty
        for key, value in self.outputs.items():
            if len(value.shape) >= 2:
                self.outputs[key] = self.outputs[key][1:]

        # TODO: Add lsdo_dashboard processing

    def update_outputs(self, **kwargs):
        name = self.problem_name
        dirName = name + '_outputs'

        for key, value in kwargs.items():
            # Only user-specified outputs will be stored
            # Multidim. arrays will be flattened (c-major/row major) before writing to a file
            if key in self.outputs:
                # if isinstance(value, np.ndarray):
                # try:
                #     value.size == 1
                if not isinstance(value, (int, float, list)):
                    # Update output file
                    with open(dirName + '/' + key + '.out', 'a') as f:
                        np.savetxt(f, value.reshape(1, value.size))
                    # Update outputs dict
                    self.outputs[key] = np.append(
                        self.outputs[key],
                        value.reshape((1, ) + value.shape),
                        axis=0)
                # else:
                # except:
                else:
                    # Update output file
                    with open(dirName + '/' + key + '.out', 'a') as f:
                        # Avoid appending None for a failed line search
                        try:
                            np.savetxt(f, [value])
                        except:
                            np.savetxt(f, [1.])

                    # Update outputs dict
                    self.outputs[key] = np.append(self.outputs[key],
                                                  value)

                # Update outputs dict
                self.out_dict[key] = value
     

            # Raise error if user tries to update ouptput not available in default outputs
            # for an optimizer
            elif key not in self.available_outputs:
                raise ValueError(
                    'Unavailable output "{}" is passed in to be updated'
                    .format(key))
        
        # Print new summary_table row
        new_row =''
        outs = self.available_outputs
        for key in self.scalar_keys:
            if outs[key] in (int, np.int_, np.int32, np.int64):
                new_row += "%16i " % self.out_dict[key]
            elif outs[key] in (float, np.float_, np.float32, np.float64):
                new_row += "%16.6E " % self.out_dict[key]
        new_row += '\n'
        self.out_dict = {key: 0 for key in self.scalar_keys}

        with open(dirName + '/' + 'summary.out', 'a') as f:
            f.write(new_row)

    def print_results(self, **kwargs):
        self._print_results(**kwargs)

    def _print_results(self,
                       title="ModOpt final iteration summary:",
                       **kwargs):

        # TODO: Testing to verify the design variable data
        # print(
        #     np.loadtxt(self.problem_name + '_outputs/x.out') -
        #     self.outputs['x'])

        # Print modopt final iteration summary

        # title = "ModOpt final iteration summary:"

        print("\n", "\t" * 1, "=" * len(title))
        print("\t" * 1, title)
        print("\t" * 1, "=" * len(title))

        longest_key = max(self.outputs, key=len)
        max_string_length = max(7, len(longest_key))
        total_length = max_string_length + 5

        print("\t" * 1, "Problem", " " * (total_length - 7), ':',
              self.problem_name)
        print("\t" * 1, "Solver", " " * (total_length - 6), ':',
              self.solver_name)

        for key, value in self.outputs.items():
            if len(value.shape) == 1:
                print("\t" * 1, key, " " * (total_length - len(key)),
                      ':', value[-1])

        bottom_line_length = total_length + 25
        # bottom_line_length = len(title)
        print("\t", "=" * bottom_line_length)

        # Print optimization summary table

        allowed_keys = {'summary_table', 'compact_print'}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, val) for key, val in kwargs.items()
                             if key in allowed_keys)

        if self.summary_table:

            dirName = self.problem_name + '_outputs'

            with open(dirName + '/summary.out', 'r') as f:
                # lines = f.readlines()
                lines = f.read().splitlines()

            title = "modOpt summary table:"
            row_length = len(lines[0])
            line_length = max(row_length, len(title))

            print("\n")
            print("=" * line_length)
            print(title.center(line_length))
            print("=" * line_length)

            # Number of iterations including zeroth iteration
            # (after removing column label)
            total_itr = len(lines) - 1

            # Print all iterations
            if not (self.compact_print):
                print(*lines, sep="\n")

            # Print only itrs_to_print above the itr_threshold
            else:
                itr_threshold = 20
                itrs_to_print = 10

                if total_itr <= itr_threshold:
                    print(*lines, sep="\n")

                else:
                    idx = np.linspace(0,
                                      total_itr - 1,
                                      itrs_to_print,
                                      dtype='int')

                    # Account for the column label index
                    idx = np.append(0, idx + 1)

                    # Lines to print
                    compact_lines = [lines[i] for i in idx]
                    print(*compact_lines, sep="\n")

            print("=" * line_length)

    def check_first_derivatives(self, x, formulation='rs'):
        obj = self.obj
        grad = self.grad

        if self.problem.ny == 0:
            nx = self.problem.nx
            nc = self.problem.nc

        ###############################
        # Only for the SURF algorithm #
        ###############################
        else:
            nx = self.problem.nx + self.problem.ny
            nc = self.problem.nc + self.problem.nr
        ###############################

        constrained = False
        if nc != 0:
            constrained = True
            con = self.con
            jac = self.jac

        ###############################
        # Only for the SURF algorithm #
        ###############################
        if formulation in ('cfs', 'surf'):
            print("INSIDE cfs or surf")
            y = self.problem.solve_residual_equations(
                x[:self.problem.nx])
            x[self.problem.nx:] = y

            self.problem.formulation = 'fs'
        ###############################

        grad_exact = grad(x)
        if constrained:
            jac_exact = jac(x)

        h = 1e-6

        grad_fd = np.full((nx, ), -obj(x), dtype=float)
        if constrained:
            jac_fd = np.outer(-con(x), np.ones((nx, ), dtype=float))

        for i in range(nx):
            e = h * np.identity(nx)[i]

            grad_fd[i] += obj(x + e)
            if constrained:
                jac_fd[:, i] += con(x + e)

        grad_fd /= h
        if constrained:
            jac_fd /= h

        EPSILON = 1e-10

        # print('grad_exact:', grad_exact)
        # print('grad_fd:', grad_fd)
        # print('jac_exact:', jac_exact)
        # print('jac_fd:', jac_fd)

        grad_abs_error = np.absolute(grad_fd - grad_exact)

        # FD is assumed to give the actual gradient
        grad_rel_error = grad_abs_error / (np.linalg.norm(grad_fd, 2) + EPSILON)
        # grad_rel_error = grad_abs_error / (np.absolute(grad_fd) + EPSILON)

        if constrained:
            jac_abs_error = np.absolute(jac_fd - jac_exact)
            jac_rel_error = jac_abs_error / (np.linalg.norm(jac_fd, 'fro') + EPSILON)

            # jac_rel_error = jac_abs_error / (np.absolute(jac_fd) + EPSILON)
            # jac_rel_error = jac_abs_error / (np.absolute(jac_exact.toarray()) + EPSILON)

        # out_buffer = StringIO()

        header = "{0} | {1} | {2} | {3} | {4} "\
                            .format(
                                pad_name('Derivative type', 8, quotes=False),
                                pad_name('Calc norm', 10),
                                pad_name('FD norm', 10),
                                pad_name('Abs error norm', 10),
                                pad_name('Rel error norm', 10),
                            )

        # out_buffer.write('\n' + header + '\n')
        print('\n' + '-' * len(header))
        print(header)

        # out_buffer.write('-' * len(header) + '\n' + '\n')
        print('-' * len(header) + '\n')

        deriv_line = "{0} | {1:.4e} | {2:.4e} | {3:.4e}     | {4:.4e}    "
        grad_line = deriv_line.format(
            pad_name('Gradient', 15, quotes=False),
            np.linalg.norm(grad_exact),
            np.linalg.norm(grad_fd),
            np.linalg.norm(grad_abs_error),
            np.linalg.norm(grad_rel_error),
        )

        # out_buffer.write(grad_line + '\n')
        print(grad_line)

        if constrained:
            if isinstance(jac_exact, np.ndarray):
                jac_exact_norm = np.linalg.norm(jac_exact)
            else:
                jac_exact_norm = sp.sparse.linalg.norm(jac_exact)

            jac_line = deriv_line.format(
                pad_name('Jacobian', 15, quotes=False),
                jac_exact_norm,
                # np.linalg.norm(jac_exact),
                # sp.sparse.linalg.norm(jac_exact),
                np.linalg.norm(jac_fd),
                np.linalg.norm(jac_abs_error),
                np.linalg.norm(jac_rel_error),
            )

            # out_buffer.write(jac_line + '\n')
            print(jac_line)

        print('-' * len(header) + '\n')