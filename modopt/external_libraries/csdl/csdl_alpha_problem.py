import numpy as np
import scipy as sp
import warnings
from modopt import Problem as OptProblem

try:
    from csdl_alpha.experimental import PySimulator
except:
    warnings.warn("PySimulator() from 'csdl_alpha' could not be imported")

class CSDLAlphaProblem(OptProblem):
    def initialize(self, ):
        self.options.declare('problem_name', default='unnamed_problem', types=str)
        self.options.declare('simulator', types=PySimulator)

    def setup(self, ):

        # Only for csdl problems
        # (Since it is not constructed explicitly using the Problem() class)
        self.problem_name = self.options['problem_name']

        # Set problem dimensions
        sim = self.options['simulator']

        (dScaler, dLower, dUpper, dInitial), (cScaler, cLower, cUpper), oScaler = sim.get_optimization_metadata()

        self.x0 = dInitial.flatten() * 1.0
        self.nx = len(self.x0)

        if not (dScaler.shape == dLower.shape == dUpper.shape == dInitial.shape == (self.nx,)):
            raise ValueError(f'Design variable metadata dimensions are not consistent. shape(dScaler, dLower, dUpper, dInitial) = ({dScaler.shape}, {dLower.shape}, {dUpper.shape}, {dInitial.shape})')
        self.x_lower  = dLower * dScaler
        self.x_upper  = dUpper * dScaler
        self.x_scaler = dScaler * 1.0

        if (cScaler is None) or (cLower is None) or (cUpper is None):
            self.nc = 0
            if (cScaler is not None) or (cLower is not None) or (cUpper is not None):
                raise ValueError(f'Constraint metadata is inconsistent. (cScaler, cLower, cUpper) = ({cScaler}, {cLower}, {cUpper})')
        else:
            self.nc = len(cScaler)
            if not (cScaler.shape == cLower.shape == cUpper.shape == (self.nc,)):
                raise ValueError(f'Constraint metadata dimensions are not consistent. shape(cScaler, cLower, cUpper) = ({cScaler.shape}, {cLower.shape}, {cUpper.shape})')
            self.c_lower  = cLower * cScaler
            self.c_upper  = cUpper * cScaler
            self.c_scaler = cScaler * 1.0
        
        if oScaler is not None:
            if not (oScaler.shape == (1,)):
                raise ValueError(f'Objective scaler dimensions must be (1,) for single objective optimization. shape(oScaler) = {oScaler.shape}')
        self.f_scaler = oScaler * 1.0 if oScaler is not None else 1.0
        
        self.model_evals = 0                      # number of model evaluations
        self.deriv_evals = 0                      # number of derivative evaluations
        self.fail1 = False                      # failure of functions
        self.fail2 = False                      # failure of functions or derivatives
        self.warm_x         = self.x0 - 1.      # (x0 - 1.) to keep it different from initial dv values
        self.warm_x_deriv   = self.x0 - 2.      # (x0 - 2.)to keep it different from initial dv and warm_x values
        if self.nc > 0:
            self.constrained = True

        self.SURF_mode = False # True if using SURF, False if using RS/FS

    def raise_issues_with_user_setup(self, ):
        pass
    
# Try except to get the failures from the simulator, but not for the first run
# If the first run fails, raise an exception
# If the subsequent runs fail, set the fail flag to True
# check_failure, guess_dict, tol_dict, force_rerun are not used in the current implementation
    
    def check_if_warm_and_run_model(self, x, guess_dict=None, tol_dict=None, 
                                    force_rerun=False, check_failure=False):
        '''
        Input x is the unscaled design variable vector.
        '''

        sim = self.options['simulator']
        if not self.SURF_mode:      # for pure RS/FS 
            if (not np.array_equal(self.warm_x, x)) or force_rerun:
                sim.update_design_variables(x)
                try:
                    f_us, c_us = sim.run_forward()
                    self.fail1 = False
                    if f_us is not None:
                        if np.isnan(np.sum(f_us)) or np.isinf(np.sum(f_us)):
                            self.fail1 = True
                            if self.model_evals == 0:
                                raise Exception('Objective returned NAN/INF for the first run. Please check the model setup.')
                        self.f_s = f_us * self.f_scaler
                            
                    if c_us is not None:
                        if np.isnan(np.sum(c_us)) or np.isinf(np.sum(c_us)):
                            self.fail1 = True
                            if self.model_evals == 0:
                                raise Exception('Constraints contain NAN/INF for the first run. Please check the model setup.')
                        self.c_s = c_us * self.c_scaler

                except:
                    self.fail1 = True
                    if self.model_evals == 0:
                        raise Exception('Model evaluation failed for the first run. Please check the model setup.')
                
                self.model_evals += 1
                self.warm_x[:] = x

            return
                
    def check_if_warm_and_compute_derivatives(self, x, guess_dict=None, tol_dict=None, 
                                              force_rerun=False, check_failure=False):
        sim = self.options['simulator']
        if not self.SURF_mode:      # for pure RS/FS                     
            if not np.array_equal(self.warm_x_deriv, x) or force_rerun:
                self.check_if_warm_and_run_model(x, force_rerun=force_rerun, check_failure=check_failure)
                if not self.fail1:
                    try:
                        g_us, j_us = sim.compute_optimization_derivatives()
                        self.fail2 = False
                        if g_us is not None:
                            if np.isnan(np.sum(g_us)) or np.isinf(np.sum(g_us)):
                                self.fail2 = True
                                if self.deriv_evals == 0:
                                    raise Exception('Objective gradient contains NAN/INF for the first run. Please check the model setup.')
                            self.g_s = g_us[0] * self.f_scaler / self.x_scaler # g_us is 2d matrix of gradients for each objective
                        
                        if j_us is not None:
                            if np.isnan(np.sum(j_us)) or np.isinf(np.sum(j_us)):
                                self.fail2 = True
                                if self.deriv_evals == 0:
                                    raise Exception('Constraint Jacobian contains NAN/INF for the first run. Please check the model setup.')
                            self.j_s = j_us * np.outer(self.c_scaler, 1./self.x_scaler)
                            
                    except:
                        self.fail2 = True
                        if self.deriv_evals == 0:
                            raise Exception('Derivative evaluation failed for the first run. Please check the model setup.')
                else:
                    self.fail2 = True
                        
                self.deriv_evals += 1
                self.warm_x_deriv[:] = x
            return

    def _setup_bounds(self): # x and c bounds don't include states y and residuals R for SURF
        pass
    def _setup_bounds(self): # x and c bounds don't include states y and residuals R for SURF
        pass

    def _setup_scalers(self):
        pass

    def compute_objective(self, dvs, obj):
        pass
    def compute_objective_gradient(self, dvs, grad):
        pass
    def compute_constraints(self, dvs, con):
        pass
    def compute_constraint_jacobian(self, dvs, jac):
        pass
    
    # TODO: Add decorators for checking if x is warm and for updating dvs
    def _compute_objective(self, x, guess_dict=None, tol_dict=None, 
                           force_rerun=False, check_failure=False):
        print('Computing objective >>>>>>>>>>')
        self.check_if_warm_and_run_model(x/self.x_scaler, guess_dict, tol_dict, 
                                         force_rerun, check_failure)
        print('---------Computed objective---------')
        return self._get_objective()
        # return failure_flag, sim.objective()

    def _compute_objective_gradient(self, x, guess_dict=None, tol_dict=None, 
                                    force_rerun=False, check_failure=False):
        print('Computing gradient >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x/self.x_scaler, guess_dict, tol_dict, 
                                                   force_rerun, check_failure)
        print('---------Computed gradient---------')
        return self._get_objective_gradient()

    def _compute_constraints(self, x, guess_dict=None, tol_dict=None, 
                             force_rerun=False, check_failure=False):
        print('Computing constraints >>>>>>>>>>')
        self.check_if_warm_and_run_model(x/self.x_scaler, guess_dict, tol_dict, 
                                         force_rerun, check_failure)
        print('---------Computed constraints---------')
        return self._get_constraints()

    def _compute_constraint_jacobian(self, x, guess_dict=None, tol_dict=None, 
                                     force_rerun=False, check_failure=False):
        print('Computing Jacobian >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x/self.x_scaler, guess_dict, tol_dict, 
                                                   force_rerun, check_failure)
        print('---------Computed Jacobian---------')
        return self._get_constraint_jacobian()

    def _compute_all(self, x, force_rerun=False, check_failure=False):                              # only for SNOPTC, (NOT meant for SURF)
        print('Computing all at once >>>>>>>>>>')
        # self.check_if_warm_and_run_model(x/self.x_scaler, force_rerun=force_rerun, check_failure=check_failure)                 # This is rqd, o/w warm derivs skip model evals --> not sure since the warm_x and warm_x_deriv are always equal for compute_all
        self.check_if_warm_and_compute_derivatives(x/self.x_scaler, force_rerun=force_rerun, check_failure=check_failure)
        print('---------Computed all at once---------')
        return self.fail2, self.f_s, self.c_s, self.g_s, self.j_s
    



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
    
    def _get_objective(self, ):
        return self.f_s[0]
    
    def _get_objective_gradient(self,):
        sim = self.options['simulator']
        if not(self.SURF_mode):
            return self.g_s
        else:
            pFpx, pCpx, pFpy, pCpy, pRpx, pRpy = sim.get_SURF_derivatives()
            return np.concatenate([pFpx]+[pFpy[key].flatten() for key in self.state_names])
    
    def _get_constraints(self,):
        sim = self.options['simulator']
        if not(self.SURF_mode):
            return self.c_s
        else:
            c = sim.constraints()
            r = np.concatenate([sim[key].flatten() for key in self.res_names])
            return np.concatenate([c, r])
            # return np.concatenate([c, r, -r])
        
    def _get_constraint_jacobian(self,):
        sim = self.options['simulator']
        if not(self.SURF_mode):
            return self.j_s
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