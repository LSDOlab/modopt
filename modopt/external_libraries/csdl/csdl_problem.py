import numpy as np
import scipy as sp
import warnings
from modopt import Problem as OptProblem
from modopt.core.recording_and_hotstart import hot_start, record

class CSDLProblem(OptProblem):
    def initialize(self, ):
        try:
            from python_csdl_backend import Simulator
        except ImportError:
            raise ImportError("Simulator() from 'python_csdl_backend' could not be imported")

        self.options.declare('problem_name', default='unnamed_problem', types=str)
        self.options.declare('simulator', types=Simulator)

    def setup(self, ):

        # Only for csdl problems
        # (Since it is not constructed explicitly using the Problem() class)
        self.problem_name = self.options['problem_name']

        # Set problem dimensions
        sim = self.options['simulator']
        try: # the fixed ordering used for state names in vectors comes from self.hstate_to_res dict
            self.hstate_to_res = sim.get_hybrid_state_and_residual_names() # dict is ordered if Python >=3.7
            self.SURF_mode = True if self.hstate_to_res else False
            if self.SURF_mode:
                self.state_names     = list(self.hstate_to_res.keys())
                self.res_names       = list(self.hstate_to_res.values())
                self.num_states      = len(self.state_names)
                self.ny = sum([sim[state_name].size for state_name in self.state_names])
                self.nr = self.ny
                self.y0 = self._get_hybrid_state_vector()
                from array_manager.api import VectorComponentsDict, Vector, MatrixComponentsDict, Matrix, CSCMatrix
                self.y_dict = VectorComponentsDict()
                for state_name in self.state_names: self.y_dict[state_name] = dict(shape=(sim[state_name].size,))
                self.y = Vector(self.y_dict)
                self.y.allocate(data=self.y0, setup_views=True)
                self.res = Vector(self.y_dict)
                self.res.allocate(data=np.ones((self.nr,)), setup_views=True)
                self.warm_tol_dict   = {}
                self.warm_guess_dict = {}
                self.tol_dict0       = dict([(state_name, 1e-12) for state_name in self.state_names])
                self.guess_dict0     = dict([(state_name, sim[state_name].flatten()) for state_name in self.state_names])
        except:
            self.SURF_mode = False
            warnings.warn('This version of CSDL or python_csdl_backend does not support SURF paradigm.')

        self.x0 = sim.design_variables()
        self.model_evals = 0                      # failure of functions
        self.deriv_evals = 0                      # failure of functions
        self.fail1 = False                      # failure of functions
        self.fail2 = False                      # failure of functions or derivatives
        self.warm_x         = self.x0 - 1.      # (x0 - 1.) to keep it differernt from initial dv values
        self.warm_x_deriv   = self.x0 - 2.      # (x0 - 2.)to keep it differernt from initial dv and warm_x values
        self.nx = len(self.x0)
        self.nc = len(sim.constraints())        # TODO: will sim.constraints() before sim.run() work?
        self.declared_variables = ['dv', 'obj', 'grad']
        self.o_scaler = self.x_scaler = 1.0
        self.c_scaler = None
        if self.nc > 0:
            self.constrained = True
            self.declared_variables += ['con', 'jac']
            self.c_scaler = 1.0

    def setup_derivatives(self):
        pass

    def _setup_scalers(self, ):
        pass

    def raise_issues_with_user_setup(self, ):
        pass

    def check_if_smaller_tol(self, tol_dict):   # only for SURF    
        if not(self.warm_tol_dict):
            return True
        for state_name, warm_tol in self.warm_tol_dict.items():
            if tol_dict[state_name] < warm_tol:
                return True
        return False
    
    def check_if_warm_and_run_model(self, x, guess_dict=None, tol_dict=None, 
                                    force_rerun=False, check_failure=False):
        sim = self.options['simulator']
        if not self.SURF_mode:      # for pure RS/FS 
            if (not np.array_equal(self.warm_x, x)) or force_rerun:
                sim.update_design_variables(x)
                self.fail1 = bool(sim.run(check_failure=check_failure))
                self.model_evals += 1
                self.warm_x[:] = x
            return
        else:                       # only for SURF
            if (not np.array_equal(self.warm_x, x)) or (self.check_if_smaller_tol(tol_dict)) or force_rerun:
                sim.update_design_variables(x)
                self._update_implicit_guess_and_tol(guess_dict, tol_dict)  # move this to sim (For Mark)
                for state_name in self.state_names: 
                    self.warm_guess_dict[state_name] = guess_dict[state_name]
                    self.warm_tol_dict[state_name]   = tol_dict[state_name]
                self.fail1 = bool(sim.run(check_failure=check_failure))
                self.model_evals += 1
                self.warm_x[:] = x
            return
                
    def check_if_warm_and_compute_derivatives(self, x, guess_dict=None, tol_dict=None, 
                                              force_rerun=False, check_failure=False):
        sim = self.options['simulator']
        if not self.SURF_mode:      # for pure RS/FS                     
            if not np.array_equal(self.warm_x_deriv, x) or force_rerun:
                self.check_if_warm_and_run_model(x, force_rerun=force_rerun, check_failure=check_failure)
                f2 = sim.compute_total_derivatives(check_failure=check_failure)
                self.deriv_evals += 1
                self.fail2 = (self.fail1 and bool(f2))
                self.warm_x_deriv[:] = x
            return
        else:                       # only for SURF
            if (not np.array_equal(self.warm_x_deriv, x)) or (self.check_if_smaller_tol(tol_dict)) or force_rerun:
                self.check_if_warm_and_run_model(x, guess_dict, tol_dict, force_rerun, check_failure)
                f2 = sim.compute_SURF_derivatives(check_failure=check_failure)
                self.deriv_evals += 1
                self.fail2 = (self.fail1 and bool(f2))
                self.warm_x_deriv[:] = x
            return

    def _setup_bounds(self): # x and c bounds don't include states y and residuals R for SURF
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

        # if self.SURF_mode:
        #     self.x_lower = np.concatenate((self.x_lower, np.full((self.ny,), -np.inf)))
        #     self.x_upper = np.concatenate((self.x_upper, np.full((self.ny,),  np.inf)))

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

        # if self.SURF_mode:
        #     # For R(x,y) >= 0
        #     self.c_lower = np.concatenate((self.c_lower, np.full((self.ny,), 0.)))
        #     self.c_upper = np.concatenate((self.c_upper, np.full((self.ny,), np.inf)))
        #     # For R(x,y) <= 0
        #     self.c_lower = np.concatenate((self.c_lower, np.full((self.ny,), -np.inf)))
        #     self.c_upper = np.concatenate((self.c_upper, np.full((self.ny,), 0.)))

    def compute_objective(self, dvs, obj):
        pass
    def compute_objective_gradient(self, dvs, grad):
        pass
    def compute_constraints(self, dvs, con):
        pass
    def compute_constraint_jacobian(self, dvs, jac):
        pass
    
    # TODO: Add decorators for checking if x is warm and for updating dvs
    @record(['x'], ['obj'])
    @hot_start(['x'], ['obj'])
    def _compute_objective(self, x, guess_dict=None, tol_dict=None, 
                           force_rerun=False, check_failure=False):
        sim = self.options['simulator']
        # print('Computing objective >>>>>>>>>>')
        self.check_if_warm_and_run_model(x, guess_dict, tol_dict, 
                                         force_rerun, check_failure)
        # print('---------Computed objective---------')
        return sim.objective()[0]
        # return failure_flag, sim.objective()

    @record(['x'], ['grad'])
    @hot_start(['x'], ['grad'])
    def _compute_objective_gradient(self, x, guess_dict=None, tol_dict=None, 
                                    force_rerun=False, check_failure=False):
        sim = self.options['simulator']
        # print('Computing gradient >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x, guess_dict, tol_dict, 
                                                   force_rerun, check_failure)
        # print('---------Computed gradient---------')
        return self._get_objective_gradient()

    @record(['x'], ['con'])
    @hot_start(['x'], ['con'])
    def _compute_constraints(self, x, guess_dict=None, tol_dict=None, 
                             force_rerun=False, check_failure=False):
        sim = self.options['simulator']
        # print('Computing constraints >>>>>>>>>>')
        self.check_if_warm_and_run_model(x, guess_dict, tol_dict, 
                                         force_rerun, check_failure)
        # print('---------Computed constraints---------')
        return self._get_constraints()

    @record(['x'], ['jac'])
    @hot_start(['x'], ['jac'])
    def _compute_constraint_jacobian(self, x, guess_dict=None, tol_dict=None, 
                                     force_rerun=False, check_failure=False):
        sim = self.options['simulator']
        # print('Computing Jacobian >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x, guess_dict, tol_dict, 
                                                   force_rerun, check_failure)
        # print('---------Computed Jacobian---------')
        return self._get_constraint_jacobian()

    @record(['x'], ['failure', 'obj', 'con', 'grad', 'jac'])
    @hot_start(['x'], ['failure', 'obj', 'con', 'grad', 'jac'])
    def _compute_all(self, x, force_rerun=False, check_failure=False):                              # only for SNOPTC, (NOT meant for SURF)
        sim = self.options['simulator']
        # print('Computing all at once >>>>>>>>>>')
        self.check_if_warm_and_run_model(x, force_rerun=force_rerun, check_failure=check_failure)                 # This is rqd, ow warm derivs skip model evals
        self.check_if_warm_and_compute_derivatives(x, force_rerun=force_rerun, check_failure=check_failure)
        # print('---------Computed all at once---------')
        return self.fail2, sim.objective(), sim.constraints(), sim.objective_gradient(), sim.constraint_jacobian()

    def _solve_hybrid_residual_equations(self, x, guess_dict, tol_dict, force_rerun=False):
        sim = self.options['simulator']
        print('Solving for hybrid states >>>>>>>>>>')
        self.check_if_warm_and_run_model(x,guess_dict, tol_dict, force_rerun, False)
        print('---------Computed hybrid states---------')
        return self._get_hybrid_state_vector()

    def _get_hybrid_state_vector(self, ):
        sim = self.options['simulator']
        return np.concatenate([sim[key].flatten() for key in self.state_names])
    
    def _get_pF_py(self, ):
        sim = self.options['simulator']
        pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
        return np.concatenate([pFpy[key].flatten() for key in self.state_names])
    
    def _get_pC_py(self, ):
        sim = self.options['simulator']
        pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
        return np.concatenate([pCpy[key] for key in self.state_names], axis=1)
    
    def _get_residuals(self,):
        sim = self.options['simulator']
        return np.concatenate([sim[key].flatten() for key in self.res_names])
    
    def _get_objective_gradient(self,):
        sim = self.options['simulator']
        if not(self.SURF_mode):
            return sim.objective_gradient()
        else:
            pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
            return np.concatenate([pFpx]+[pFpy[key].flatten() for key in self.state_names])
    
    def _get_constraints(self,):
        sim = self.options['simulator']
        if not(self.SURF_mode):
            return sim.constraints()
        else:
            c = sim.constraints()
            r = np.concatenate([sim[key].flatten() for key in self.res_names])
            return np.concatenate([c, r])
            # return np.concatenate([c, r, -r])
        
    def _get_constraint_jacobian(self,):
        sim = self.options['simulator']
        if not(self.SURF_mode):
            return sim.constraint_jacobian()
        else:
            pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
            pCpy_jac = np.concatenate([pCpy[key] for key in self.state_names], axis=1)
            pRpx_jac = np.concatenate([pRpx[key] for key in self.state_names], axis=0)
            pRpy_jac = sp.linalg.block_diag(*[pRpy[key] for key in self.state_names])
            jac = np.block([[pCpx, pCpy_jac], [pRpx_jac, pRpy_jac]])
            # jac = np.block([[pCpx, pCpy_jac], [pRpx_jac, pRpy_jac], [-pRpx_jac, -pRpy_jac]])
            return jac

    # def _compute_surf_adjoint(self, x, lag, guess_dict, tol_dict):   # lag is constraint Lag.mults., only for SURF
    #     sim = self.options['simulator']

    #     print('Computing Derivatives for Adjoint >>>>>>>>>>')
    #     self.check_if_warm_and_compute_derivatives(x, guess_dict, tol_dict)
    #     print('---------Computed Derivatives for Adjoint---------')

    #     print('Computing Adjoint >>>>>>>>>>')
    #     pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
    #     # adj = {}
    #     adj_vec = np.array([])
    #     for state_name in self.state_names:
    #         pCpy_SURF = 
    #         rhs = pFpy[state_name].flatten() + pCpy[state_name].T @ lag
    #         # adj[state_name] = -np.linalg.solve(pRpy[state_name].T, rhs)
    #         adj_vec = np.append(adj_vec, -np.linalg.solve(pRpy[state_name].T, rhs))
    #     print('---------Computed Adjoint---------')
        
    #     return adj_vec
        
        # prob_name = self.options['problem_name']
        # if prob_name == 'WFopt':
        #     adj = {}
        #     for state_name in self.state_names:
        #         adj[state_name] = -np.linalg.solve(pRpy[state_name].T, vec[state_name])
        # elif prob_name == 'Trim':
        #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        # elif prob_name == 'motor':
        #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        # elif prob_name == 'BEM':
        #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        # elif prob_name == 'Ozone':
        #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')
        # else:
        #     raise NotImplementedError(f'Adjoint computation is not implemented for {prob_name}.')

    
    def _update_implicit_guess_and_tol(self, guess_dict, tol_dict):
        sim = self.options['simulator']
        for state_name in self.state_names:
            sim.set_implicit_guess_and_tol(state_name, guess_dict[state_name], tol_dict[state_name])

    def reset_eval_counts(self, ):
        self.model_evals = 0
        self.deriv_evals = 0