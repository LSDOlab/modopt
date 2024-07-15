import numpy as np
import scipy as sp
import os, shutil, copy
from datetime import datetime
import contextlib

from modopt.utils.options_dictionary import OptionsDictionary
from modopt.utils.general_utils import pad_name
# from io import StringIO
from modopt.core.problem import Problem
from modopt.core.problem_lite import ProblemLite
import warnings

class Optimizer(object):
    def __init__(self, problem, **kwargs):

        self.options = OptionsDictionary()
        self.problem = problem
        self.problem_name = problem.problem_name
        self.solver_name = 'unnamed_solver'
        self.options.declare('formulation', default='rs', types=str)

        self.initialize()
        self.options.update(kwargs)
        self._setup()

    def _setup(self):
        # Setup outputs to be written to file
        self.setup_outputs()
        # User defined optimizer-specific setup
        self.setup()

    def setup_outputs(self):
        '''
        Set up the files to write the outputs of the optimization problem.
        Three different types of outputs are written:
            1. Summary table: Contains the scalar outputs of the optimization problem
            2. Readable outputs: Contains the declared readable outputs
            3. Recorder outputs: Contains all the outputs of the optimization problem, if recording is enabled
        '''
        now       = datetime.now()
        timestamp = self.timestamp = now.strftime("%Y-%m-%d_%H.%M.%S.%f")
        dir       = self.out_dir = f"{self.problem_name}_outputs/{timestamp}"
        a_outs    = self.available_outputs             # Available outputs dictionary
        d_outs    = self.options['readable_outputs']   # Declared outputs list

        self.scalar_outputs = [out for out in a_outs.keys() if not isinstance(a_outs[out], tuple)]
        s_outs = self.scalar_outputs

        # Create the outputs directory
        # if len(s_outs) > 0 or len(d_outs) > 0 or self.options['recording']:        
        os.makedirs(dir)

        # 1. Write the header of the summary_table file
        if len(s_outs) > 0:
            header =''
            for key in s_outs:
                if a_outs[key] in (int, np.int_, np.int32, np.int64):
                    header += "%10s " % key
                elif a_outs[key] in (float, np.float_, np.float32, np.float64):
                    header += "%16s " % key

            with open(f"{dir}/modOpt_summary.out", 'w') as f:
                f.write(header)

        # 2. Create the readable output files
        for key in d_outs:
            if key not in a_outs:
                raise ValueError(f'Invalid readable output "{key}" is declared.' \
                                 f'Available outputs are {list(a_outs.keys())}.')
            with open(f"{dir}/{key}.out", 'w') as f:
                pass

    def setup(self, ):
        pass

    def run_post_processing(self):
        '''
        Run the post-processing functions of the optimizer.
        1. Print the results of the optimization problem
        2. Write the outputs to the corresponding files
        3. Run the lsdo_dashboard processing
        '''
        with open(f"{self.out_dir}/modOpt_results.out", 'w') as f:
            with contextlib.redirect_stdout(f):
                self.print_results(all=True)
        # TODO: Add lsdo_dashboard processing

    def update_outputs(self, **kwargs):
        '''
        Update and write the outputs of the optimization problem to the corresponding files.
        Three different types of outputs are written:
            1. Summary table: Contains the scalar outputs of the optimization problem
            2. Readable outputs: Contains the declared readable outputs
            3. Recorder outputs: Contains all the outputs of the optimization problem, if recording is enabled
        '''
        dir    = self.out_dir
        a_outs = self.available_outputs             # Available outputs dictionary
        d_outs = self.options['readable_outputs']   # Declared outputs list

        if set(kwargs.keys()) != set(a_outs):
            raise ValueError(f'Output(s) passed in to be updated {list(kwargs.keys())} ' \
                             f'do not match the available outputs {list(a_outs.keys())}.')
        
        # 1. Write the scalar outputs to the summary file
        if len(self.scalar_outputs) > 0:
            # Print summary_table row
            new_row ='\n'
            for key in self.scalar_outputs:
                if a_outs[key] in (int, np.int_, np.int32, np.int64):
                    new_row += "%10i " % kwargs[key]
                elif a_outs[key] in (float, np.float_, np.float32, np.float64):
                    new_row += "%16.6E " % kwargs[key]

            with open(f"{dir}/modOpt_summary.out", 'a') as f:
                f.write(new_row)

        # 2. Write the declared readable outputs to the corresponding files
        for key in d_outs:
            value = kwargs[key]
            if key in self.scalar_outputs:
                if np.isscalar(value) and np.isreal(value):
                    with open(f"{dir}/{key}.out", 'a') as f:
                        np.savetxt(f, [value])
                else:
                    raise ValueError(f'Value of "{key}" is not a real-valued scalar.')        
            else:
                # Multidim. arrays will be flattened (c-major/row major) before writing to a file
                with open(f"{dir}/{key}.out", 'a') as f:
                    np.savetxt(f, value.reshape(1, value.size))
        
        # 3. TODO: Write the outputs to the recording files
        self.out_dict = out_dict = copy.deepcopy(kwargs)

    def check_if_callbacks_are_declared(self, cb, cb_str, solver_str):
        if cb not in self.problem.user_defined_callbacks:
            if isinstance(self.problem, Problem):
                raise ValueError(f"{cb_str} function is not declared in the Problem() subclass but is needed for {solver_str}.")
            elif isinstance(self.problem, ProblemLite):
                warnings.warn(f"{cb_str} function is not provided in the ProblemLite() container but is needed for {solver_str}. "\
                              f"The optimizer will use finite differences to compute the {cb_str}.")

    def print_results(self, summary_table=False, all=False):

        # TODO: Testing to verify the design variable data
        # print(
        #     np.loadtxt(self.problem_name + '_outputs/x.out'))

        output  = "\n\tSolution from modOpt:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':25}: {self.problem_name}"
        output += f"\n\t{'Solver':25}: {self.solver_name}"
        for key, value in self.results.items():
            if np.isscalar(value):
                output += f"\n\t{key:25}: {value}"

        output += '\n\t' + '-'*100
        print(output)

        if summary_table:
            with open(f"{self.out_dir}/modOpt_summary.out", 'r') as f:
                # lines = f.readlines()
                lines = f.read().splitlines()

            title = "modOpt summary table:"
            line_length = max(len(lines[0]), len(title))

            # Print header
            output  = "\n" + "=" * line_length
            output += f"\n{title.center(line_length)}"
            output += "\n" + "=" * line_length

            # Print all iterations
            output += "\n" + "\n".join(lines)

            output += "\n" + "=" * line_length
            print(output)


    def check_first_derivatives(self, x=None, step=1e-6, formulation='rs'):
        obj = self.obj
        grad = self.grad

        if x is None:
            x = self.problem.x0

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

        h = step

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