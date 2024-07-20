import warnings
import numpy as np
import time

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not found, plotting disabled.")
    plt = None

class Visualizer:

    def __init__(self, problem_name, visualize_vars, out_dir):
        '''
        Initialize the visualizer with the scalar variables to visualize. 
        The variables should be a list of strings, where each string is the name of a variable to visualize. 
        Some examples for the variables are as follows (availability depends on the optimizer used):
            - 'obj'         : the objective function value
            - 'opt'         : the optimality measure
            - 'feas'        : the feasibility measure
            - 'x[i]'        : the i-th variable value
            - 'con[i]'      : the i-th constraint value
            - 'jac[i,j]'    : the (i,j)-th element of the Jacobian matrix
            - 'grad[i]'     : the i-th gradient value
            - 'lmult[i]'    : the i-th Lagrange multiplier value

        Creates an interactive plot with the specified variables on the y-axis and the iteration number on the x-axis.
        The plots are stacked vertically in the order they are specified in the list.
        The plot is updated with the latest values of the variables after each call to the update_plot method.

        Parameters
        ----------
        problem_name : str
            Name of the optimization problem.
        visualize_vars : list of str
            List of variables to visualize.
        out_dir : str
            Path to the directory where the visualization will be saved.
        '''

        v_start = time.time()
        if plt is None:
            raise ImportError("matplotlib not found, cannot visualize.")
        self.problem_name   = problem_name
        self.visualize_vars = visualize_vars
        self.save_figname   = out_dir + '/visualization.pdf'
        plt.ion()
        lines_dict = {}
        var_dict = {}
        n_plots = len(visualize_vars)
        self.fig, self.axs = plt.subplots(n_plots, figsize=(10, 3*n_plots))
        self.fig.suptitle(f'modOpt Optimization visualization [{problem_name}]')
        for ax, var in zip(self.axs, visualize_vars):
            # ax.set_title(var)
            # ax.set_xlabel('Iteration')
            ax.set_ylabel(var)
            var_dict[var] = []
            # add any semilogy variables here (when new optimizers are added to modOpt)
            if var in ['opt', 'feas', 'rho', 'merit', 'f_sd', 'tr_radius', 'constr_penalty', 'barrier_parameter', 'barrier_tolerance']:
                lines_dict[var], = ax.semilogy([], [], label=var)
            else:
                lines_dict[var], = ax.plot([], [], label=var)
            ax.legend()

        # self.fig.set_figwidth(8)
        # self.fig.set_figheight(3*n_plots)
        # self.fig.set_size_inches(10, 3*len(self.visualize_vars), forward=True)
        # plt.gcf().set_size_inches(10, 3*len(self.visualize_vars))
        plt.tight_layout(pad=3.0, h_pad=0.1, w_pad=0.1, rect=[0, 0, 1., 1.])

        self.lines_dict = lines_dict
        self.var_dict   = var_dict

        self.vis_time  = time.time() - v_start
        self.wait_time = 0.0

    def update_plot(self, out_dict):
        '''
        Update the plot with the latest values of variables provided in the out_dict.
        Appends the values of scalar iterates after each iteration in the var_dict attribute before updating the plot.
        The out_dict should be a dictionary containing the variable names as keys.
        Some examples for the variables are as follows (depends on the optimizer used):
            - 'obj'     : the objective function value
            - 'opt'     : the optimality condition
            - 'feas'    : the feasibility condition
            - 'x'       : the variable values
            - 'con'     : the constraint values
            - 'jac'     : the Jacobian matrix
            - 'grad'    : the gradient values
            - 'lmult'   : the Lagrange multiplier values
        '''

        v_start = time.time()
        for k, s_var in enumerate(self.visualize_vars):
            var   = s_var.split('[')[0]
            if var in out_dict:
                if '[' not in s_var:
                    self.var_dict[s_var].append(out_dict[var] * 1.0) # *1.0 is necessary so that the value is not a reference
                elif ',' not in s_var:
                    idx = int(s_var.split('[')[1].split(']')[0])
                    self.var_dict[s_var].append(out_dict[var][idx] * 1.0)
                else:
                    idx1, idx2 = map(int, s_var.split('[')[1].split(']')[0].split(','))
                    self.var_dict[s_var].append(out_dict[var][idx1, idx2] * 1.0)
                
                x_data = np.arange(len(self.var_dict[s_var]))
                self.lines_dict[s_var].set_data(x_data, self.var_dict[s_var])
                
            # Rescale the plot
            self.axs[k].relim()
            self.axs[k].autoscale_view()

        # time.sleep(0.5)

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.vis_time += time.time() - v_start
        
    def save_plot(self, ):
        '''
        Save the plot to a file.
        '''
        v_start = time.time()
        # plt.gcf().set_size_inches(10, 3*len(self.visualize_vars))
        # self.fig.set_size_inches(10, 3*len(self.visualize_vars), forward=True)
        self.fig.savefig(self.save_figname,)
        self.vis_time += time.time() - v_start

    def close_plot(self):
        '''
        Close the plot.
        '''
        self.save_plot()
        
        plt.ioff()   
        plt.close()

    def keep_plot(self):
        '''
        Keep the plot open after the optimization is completed.
        '''
        self.save_plot()

        w_start = time.time()
        plt.ioff()
        plt.show()
        self.wait_time += time.time() - w_start