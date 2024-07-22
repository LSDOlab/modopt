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
from modopt.core.visualization import Visualizer
import warnings

try:
    import h5py
except ImportError:
    warnings.warn("h5py not found, recording disabled")

class Optimizer(object):
    def __init__(self, problem, **kwargs):

        now = datetime.now()
        self.timestamp = now.strftime("%Y-%m-%d_%H.%M.%S.%f")

        self.options = OptionsDictionary()
        self.problem = problem
        self.problem_name = problem.problem_name
        self.solver_name = 'unnamed_solver'
        self.options.declare('recording', default=False, types=bool)
        self.options.declare('hot_start_from', default=None, types=(type(None), str))
        self.options.declare('hot_start_atol', default=0., types=float)
        self.options.declare('hot_start_rtol', default=0., types=float)
        self.options.declare('visualize', default=[], types=list)
        self.options.declare('turn_off_outputs', False, types=bool)
        self.update_outputs_count = 0

        self.options.declare('formulation', default='rs', types=str)

        self.initialize()
        self.options.update(kwargs)

        # compute the scalar outputs from the optimizer after initialization
        a_outs = self.available_outputs
        self.scalar_outputs = [out for out in a_outs.keys() if not isinstance(a_outs[out], tuple)]

        # Create the outputs directory
        if not self.options['turn_off_outputs']:
            self.out_dir = f"{problem.problem_name}_outputs/{self.timestamp}"
            self.modopt_output_files  = [f"directory: {self.out_dir}", 'modopt_results.out']
            os.makedirs(self.out_dir) # recursively create the directory
        else:
            if self.options['recording']:
                raise ValueError("Cannot record with 'turn_off_outputs=True'.")
            if self.options['readable_outputs'] != []:
                raise ValueError("Cannot write 'readable_outputs' with 'turn_off_outputs=True'.")
            if self.options['visualize'] != []:
                raise ValueError("Cannot visualize with 'turn_off_outputs=True'.")

        # Hot starting and recording should start even before setup() is called
        # since there might be callbacks in the setup() function
        self.record  = self.problem._record = None      # Reset if using the same problem object again
        self.problem._callback_count        = 0         # Reset if using the same problem object again
        self.problem._reused_callback_count = 0         # Reset if using the same problem object again
        self.problem._hot_start_mode        = False     # Reset if using the same problem object again
        self.problem._hot_start_record      = None      # Reset if using the same problem object again
        self.problem._num_callbacks_found   = 0         # Reset if using the same problem object again
        self.problem._hot_start_tol         = None      # Reset if using the same problem object again
        self.problem._visualizer            = None      # Reset if using the same problem object again
        
        if self.options['recording']:
            self.record  = self.problem._record = h5py.File(f'{self.out_dir}/record.hdf5', 'a')
        if self.options['hot_start_from'] is not None:
            self.setup_hot_start()
        if self.options['visualize'] != []:
            # NOTE: This will neglect 'obj_hess', 'lag_hess' active_callbacks for IPOPT
            #       and 'obj_hess', 'lag_hess', 'obj_hvp' for TrustConstr 
            #       since these get added in the setup() function.
            self.setup_visualization()
            
        self._setup()

    def _setup(self):
        # User defined optimizer-specific setup
        self.setup()
        # Setup outputs to be written to file
        if not self.options['turn_off_outputs']:
            self.setup_outputs()

    def setup_outputs(self):
        '''
        Set up the directory and open files to write the outputs of the optimization problem.
        Four different types of outputs are written:
            1. Summary table:    Single file with the scalar outputs of the optimization problem.
            2. Readable outputs: A file for each readable_output declared.
            3. Recorder:         Contains all the outputs of the optimization problem, if recording is enabled.
            4. Results:          Single file with the readable print_results() string (no setup needed).
        '''
        dir     = self.out_dir
        a_outs  = self.available_outputs             # Available outputs dictionary
        d_outs  = self.options['readable_outputs']   # Declared outputs list
        s_outs  = self.scalar_outputs                # Scalar outputs list

        # 1. Write the header of the summary_table file
        if len(s_outs) > 0:
            header = "%10s " % '#'
            for key in s_outs:
                if a_outs[key] in (int, np.int_, np.int32, np.int64):
                    header += "%10s " % key
                elif a_outs[key] in (float, np.float_, np.float32, np.float64):
                    header += "%16s " % key

            with open(f"{dir}/modopt_summary.out", 'w') as f:
                f.write(header)
            self.modopt_output_files += ["modopt_summary.out"]

        # 2. Create the readable output files
        for key in d_outs:
            if key not in a_outs:
                raise ValueError(f'Invalid readable output "{key}" is declared.' \
                                 f'Available outputs are {list(a_outs.keys())}.')
            with open(f"{dir}/{key}.out", 'w') as f:
                pass
            self.modopt_output_files += [f"{key}.out"]

        # 3. Create the recorder output file and write the attributes
        if self.options['recording']:
            constrained = self.problem.constrained
            rec = self.record
            self.modopt_output_files += ['record.hdf5']

            rec.attrs['problem_name']   = self.problem_name
            rec.attrs['solver_name']    = self.solver_name
            rec.attrs['modopt_output_files'] = self.modopt_output_files
            if hasattr(self, 'default_solver_options'):
                solver_opts = self.solver_options.get_pure_dict()
                for key, value in solver_opts.items():
                    value = 'None' if value is None else value
                    if isinstance(value, (int, float, bool, str)):
                        rec.attrs[f'solver_options-{key}'] = value
            elif self.solver_name == 'ipopt': # ipopt-specific
                for key, value in self.nlp_options['ipopt'].items():
                    if isinstance(value, (int, float, bool, str)):
                        rec.attrs[f'solver_options-{key}'] = value
            elif self.solver_name.startswith('convex_qpsolvers'): # convex_qpsolvers-specific
                for key, value in self.options['solver_options'].items():
                    if isinstance(value, (int, float, bool, str)):
                        rec.attrs[f'solver_options-{key}'] = value
            else: # for inbuilt solvers
                opts = self.options.get_pure_dict()
                for key, value in opts.items():
                    value = 'None' if value is None else value
                    if isinstance(value, (int, float, bool, str)):
                        rec.attrs[f'options-{key}'] = value
            rec.attrs['readable_outputs'] = d_outs
            rec.attrs['recording'] = str(self.options['recording'])
            rec.attrs['hot_start_from'] = str(self.options['hot_start_from'])
            rec.attrs['visualize'] = self.options['visualize']
            rec.attrs['timestamp'] = self.timestamp
            rec.attrs['constrained'] = constrained
            rec.attrs['nx'] = self.problem.nx
            rec.attrs['nc'] = self.problem.nc

            rec.attrs['x0']       = self.problem.x0 / self.problem.x_scaler
            rec.attrs['x_scaler'] = self.problem.x_scaler
            rec.attrs['o_scaler'] = self.problem.o_scaler # Only for single-objective problems
            rec.attrs['x_lower']  = self.problem.x_lower / self.problem.x_scaler
            rec.attrs['x_upper']  = self.problem.x_upper / self.problem.x_scaler
            rec.attrs['c_scaler'] = self.problem.c_scaler if constrained else 'None'
            rec.attrs['c_lower']  = self.problem.c_lower / self.problem.c_scaler if constrained else 'None'
            rec.attrs['c_upper']  = self.problem.c_upper / self.problem.c_scaler if constrained else 'None'

    def setup_hot_start(self):
        '''
        Open the hot-start record file, compute the number of callbacks found in it,
        and pass both to the problem object.
        '''
        self.hot_start_record                 = h5py.File(self.options['hot_start_from'], 'r')
        num_callbacks_found = len([key for key in list(self.hot_start_record.keys()) if key.startswith('callback_')])
        self.problem._hot_start_mode          = True
        self.problem._hot_start_record        = self.hot_start_record
        self.problem._num_callbacks_found     = num_callbacks_found
        self.problem._hot_start_tol           = (self.options['hot_start_rtol'], self.options['hot_start_atol'])

    def setup_visualization(self,):
        '''
        Setup the visualization for scalar variables of the optimization problem.
        Variables can be either optimizer outputs or callback inputs/outputs.
        '''
        visualize_vars   = []
        available_vars  = sorted(list(set(list(self.available_outputs.keys()) + self.active_callbacks + ['x'])))
        for s_var in self.options['visualize']: # scalar variables
            var = s_var.split('[')[0]
            if var not in available_vars:
                raise ValueError(f"Unavailable variable '{var}' is declared for visualization. " \
                                 f"Available variables for visualization are {available_vars}.")
            if var in self.scalar_outputs + ['obj', 'lag']:
                if var != s_var:
                    raise ValueError(f"Scalar variable '{var}' is indexed for visualization.")
            else:
                if var == s_var:
                    raise ValueError(f"Non-scalar variable '{var}' is not indexed for visualization. " \
                                     f"Provide an index to a scalar entry in '{var}' for visualization.")
            
            if var in self.available_outputs.keys():
                visualize_vars.append(s_var)
            if var in self.active_callbacks + ['x']:
                visualize_vars.append('callback_' + s_var)
        
        # No need to visualize callbacks if all variables are optimizer outputs
        visualize_callbacks = True
        if all(s_var.split('[')[0] in self.available_outputs.keys() for s_var in self.options['visualize']):
            visualize_callbacks = False
        
        self.visualizer = Visualizer(self.problem_name, visualize_vars, self.out_dir)
        if visualize_callbacks:
            self.problem._visualizer = self.visualizer
            
    def setup(self, ):
        pass

    def run_post_processing(self):
        '''
        Run the post-processing functions of the optimizer.
        1. Write the print_results() output to the the results.out file
        2. Write self.results to the record file
        3. Save and close the visualization plot
        '''
        if self.options['turn_off_outputs']:
            return
        
        self.results['out_dir'] = self.out_dir
        with open(f"{self.out_dir}/modopt_results.out", 'w') as f:
            with contextlib.redirect_stdout(f):
                self.print_results(all=True)
        if self.options['recording']:
            self.results['total_callbacks'] = self.problem._callback_count
            group = self.record.create_group('results')
            for key, value in self.results.items():
                if self.solver_name.startswith('convex_qpsolvers') and key in ['problem', 'extras']:
                    continue
                if isinstance(value, dict):
                    for k, v in value.items():
                        group[f"{key}-{k}"] = v
                else:
                    group[key] = value
        
        if self.options['visualize'] != []:
            self.visualizer.close_plot()
            self.vis_time = self.visualizer.vis_time  

    def update_outputs(self, **kwargs):
        '''
        Update and write the outputs of the optimization problem to the corresponding files.
        Three different types of outputs are written:
            1. Summary table: Contains the scalar outputs of the optimization problem
            2. Readable outputs: Contains the declared readable outputs
            3. Recorder outputs: Contains all the outputs of the optimization problem, if recording is enabled
        '''
        if self.options['turn_off_outputs']:
            return
        
        self.out_dict = out_dict = copy.deepcopy(kwargs)
        if self.options['visualize'] != []:
            self.visualizer.update_plot(out_dict)

        dir    = self.out_dir
        a_outs = self.available_outputs             # Available outputs dictionary
        d_outs = self.options['readable_outputs']   # Declared outputs list

        if set(kwargs.keys()) != set(a_outs):
            raise ValueError(f'Output(s) passed in to be updated {list(kwargs.keys())} ' \
                             f'do not match the available outputs {list(a_outs.keys())}.')
        
        # 1. Write the scalar outputs to the summary file
        if len(self.scalar_outputs) > 0:
            # Print summary_table row
            new_row ='\n' + "%10i " % self.update_outputs_count
            for key in self.scalar_outputs:
                if a_outs[key] in (int, np.int_, np.int32, np.int64):
                    new_row += "%10i " % kwargs[key]
                elif a_outs[key] in (float, np.float_, np.float32, np.float64):
                    new_row += "%16.6E " % kwargs[key]

            with open(f"{dir}/modopt_summary.out", 'a') as f:
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
        
        # 3. Write the outputs to the recording files
        if self.options['recording']:
            group_name = 'iteration_' + str(self.update_outputs_count)
            group = self.record.create_group(group_name)
            for var, value in out_dict.items():
                group[var] = value
                
        self.update_outputs_count += 1

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
            with open(f"{self.out_dir}/modopt_summary.out", 'r') as f:
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
            con = self.problem._compute_constraints
            jac = self.problem._compute_constraint_jacobian

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