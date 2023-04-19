import numpy as np

from modopt.api import Problem as OptProblem
# from csdl_lite import Simulator
# from csdl_om import Simulator
try:
    from python_csdl_backend import Simulator
except:
    print("Warning: Simulator() from 'python_csdl_backend' could not be imported")

try:
    from csdl import Model
except:
    print("Warning: Model() from 'csdl' could not be imported")


class CSDLProblem(OptProblem):
    def initialize(self, ):
        self.options.declare('problem_name',
                             default='unnamed_problem',
                             types=str)
        self.options.declare('simulator')
        # TODO:
        # self.options.declare('simulator', types=type(Simulator()))

    def setup(self, ):

        # Only for csdl problems
        # (Since it is not constructed explicitly using the Problem() class)
        self.problem_name = self.options['problem_name']

        # Set problem dimensions
        sim = self.options['simulator']
        self.x0 = sim.design_variables()
        self.fail1 = False                      # failure of functions
        self.fail2 = False                      # failure of functions or derivatives
        self.warm_x         = self.x0 - 1.      # (x0 - 1.) to keep it differernt from initial dv values
        self.warm_x_deriv   = self.x0 - 2.      # (x0 - 2.)to keep it differernt from initial dv and warm_x values
        self.nx = len(self.x0)
        self.nc = len(sim.constraints())

    def check_if_warm_x(self, x, deriv=False):
        if not(deriv):
            return np.array_equal(self.warm_x, x)
        return np.array_equal(self.warm_x_deriv, x)

    def _setup_bounds(self):
        sim = self.options['simulator']

        # Set design variable bounds
        dv_meta = sim.get_design_variable_metadata()
        x_l = []
        x_u = []
        for key in sim.dv_keys:
            shape = sim[key].shape
            l = dv_meta[key]['lower']
            u = dv_meta[key]['upper']

            x_l = np.concatenate((x_l, (l * np.ones(shape)).flatten()))
            x_u = np.concatenate((x_u, (u * np.ones(shape)).flatten()))

        self.x_lower = np.where(x_l == -1.0e30, -np.inf, x_l)
        self.x_upper = np.where(x_u == 1.0e30, np.inf, x_u)

        # Set constraint bounds
        c_meta = sim.get_constraints_metadata()
        c_l = []
        c_u = []
        for key in sim.constraint_keys:
            shape = sim[key].shape
            e = c_meta[key]['equals']
            if e is None:
                l = c_meta[key]['lower']
                u = c_meta[key]['upper']
            else:
                l = e
                u = e

            c_l = np.concatenate((c_l, (l * np.ones(shape)).flatten()))
            c_u = np.concatenate((c_u, (u * np.ones(shape)).flatten()))

        self.c_lower = np.where(c_l == -1.0e30, -np.inf, c_l)
        self.c_upper = np.where(c_u == 1.0e30, np.inf, c_u)

    def compute_objective(self, dvs, obj):
        pass
    def compute_objective_gradient(self, dvs, grad):
        pass
    def compute_constraints(self, dvs, con):
        pass
    def compute_constraint_jacobian(self, dvs, jac):
        pass
    
    # TODO: Add decorators for checking if x is warm and for updating dvs
    def _compute_objective(self, x):
        sim = self.options['simulator']
        print('Computing objective >>>>>>>>>>')
        if not (self.check_if_warm_x(x)):
            sim.update_design_variables(x)
            self.fail1 = sim.run(check_failure=True)
            self.warm_x[:] = x
        print('---------Computed objective---------')
        return sim.objective()[0]
        # return failure_flag, sim.objective()

    def _compute_objective_gradient(self, x):
        sim = self.options['simulator']
        print('Computing gradient >>>>>>>>>>')
        if not (self.check_if_warm_x(x, deriv=True)):
            if not (self.check_if_warm_x(x,)):
                sim.update_design_variables(x)
                self.fail1 = sim.run(check_failure=True)
                self.warm_x[:] = x
            f2 = sim.compute_total_derivatives(check_failure=True)
            self.fail2 = (self.fail1 and f2)
            self.warm_x_deriv[:] = x
        print('---------Computed gradient---------')
        return sim.objective_gradient()
        # return failure_flag, sim.objective_gradient()

    def _compute_constraints(self, x):
        sim = self.options['simulator']
        print('Computing constraints >>>>>>>>>>')
        if not (self.check_if_warm_x(x)):
            sim.update_design_variables(x)
            self.fail1 = sim.run(check_failure=True)
            self.warm_x[:] = x
        print('---------Computed constraints---------')
        return sim.constraints()
        # return failure_flag, sim.constraints()

    def _compute_constraint_jacobian(self, x):
        sim = self.options['simulator']
        print('Computing Jacobian >>>>>>>>>>')
        if not (self.check_if_warm_x(x, deriv=True)):
            if not (self.check_if_warm_x(x,)):
                sim.update_design_variables(x)
                self.fail1 = sim.run(check_failure=True)
                self.warm_x[:] = x
            f2 = sim.compute_total_derivatives(check_failure=True)
            self.fail2 = (self.fail1 and f2)
            self.warm_x_deriv[:] = x
        print('---------Computed Jacobian---------')
        return sim.constraint_jacobian()
        # return failure_flag, sim.constraint_jacobian()

    def _compute_all(self, x):
        sim = self.options['simulator']
        print('Computing all at once >>>>>>>>>>')
        if not (self.check_if_warm_x(x, deriv=True) and self.check_if_warm_x(x,)):
            sim.update_design_variables(x)
            self.fail1 = sim.run(check_failure=True)
            self.warm_x[:] = x
            f2 = sim.compute_total_derivatives(check_failure=True)
            self.fail2 = (self.fail1 and f2)
            self.warm_x_deriv[:] = x

        print('---------Computed all at once---------')
        return self.fail2, sim.objective(), sim.constraints(), sim.objective_gradient(), sim.constraint_jacobian()
    
    def _compute_adjoint(self, x, vec):              # vec is pl/py
        sim = self.options['simulator']

        print('Computing Derivatives for Adjoint >>>>>>>>>>')
        if not (self.check_if_warm_x(x, deriv=True)):
            if not (self.check_if_warm_x(x,)):
                sim.update_design_variables(x)
                self.fail1 = sim.run(check_failure=True)
                self.warm_x[:] = x
            f2 = sim.compute_total_derivatives(check_failure=True)
            self.fail2 = (self.fail1 and f2)
            self.warm_x_deriv[:] = x
        print('---------Computed Derivatives for Adjoint---------')

        print('Computing Adjoint >>>>>>>>>>')
        prob_name = self.options['problem_name']
        if prob_name == 'WFLOP':
            adj = None
        elif prob_name == 'Trim':
            raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        elif prob_name == 'Motor':
            raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        elif prob_name == 'BEM':
            raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        elif prob_name == 'Ozone':
            raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        else:
            raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        
        print('---------Computed Adjoint---------')

        return sim.constraint_jacobian()
