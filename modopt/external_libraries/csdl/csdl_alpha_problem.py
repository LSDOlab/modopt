import numpy as np
import scipy as sp
from modopt import Problem as OptProblem
from modopt.core.recording_and_hotstart import hot_start, record

class CSDLAlphaProblem(OptProblem):
    def initialize(self, ):
        try:
            from csdl_alpha.backends.simulator import SimulatorBase
        except ImportError:
            raise ImportError("SimulatorBase() from 'csdl_alpha' could not be imported")

        self.options.declare('problem_name', default='unnamed_problem', types=str)
        self.options.declare('simulator', types=SimulatorBase)

    def setup(self, ):

        # Only for csdl problems
        # (Since it is not constructed explicitly using the Problem() class)
        self.problem_name = self.options['problem_name']

        # Set problem dimensions
        sim = self.options['simulator']

        (dScaler, dLower, dUpper, dInitial, dAdder), (cScaler, cLower, cUpper, cAdder), (oScaler, oAdder) = sim.get_optimization_metadata()

        self.nx = len(dInitial.flatten())

        if not (dScaler.shape == dLower.shape == dUpper.shape == dInitial.shape == dAdder.shape == (self.nx,)):
            raise ValueError(f'Design variable metadata dimensions are not consistent. shape(dvScaler, dvLower, dvUpper, dvInitial, dvAdder) = ({dScaler.shape}, {dLower.shape}, {dUpper.shape}, {dInitial.shape}, {dAdder.shape})')
        # Note that x0, x_lower, x_upper are all in scaled form
        self.x0 = (dInitial+dAdder) * dScaler
        self.x_lower  = (dLower+dAdder) * dScaler
        self.x_upper  = (dUpper+dAdder) * dScaler
        self.x_scaler = dScaler * 1.0
        self.x_adder  = dAdder * 1.0

        if (cScaler is None) or (cLower is None) or (cUpper is None) or (cAdder is None):
            self.nc = 0
            self.c_adder = None # This is required since OptProblem does not initialize c_adder=None as for c_scaler and causes issues for unconstrained problems
            if (cScaler is not None) or (cLower is not None) or (cUpper is not None) or (cAdder is not None):
                raise ValueError(f'Constraint metadata is inconsistent. (cScaler, cLower, cUpper, cAdder) = ({cScaler}, {cLower}, {cUpper}, {cAdder})')
        else:
            self.nc = len(cScaler.flatten()) # or cAdder.flatten()
            if not (cScaler.shape == cLower.shape == cUpper.shape == cAdder.shape == (self.nc,)):
                raise ValueError(f'Constraint metadata dimensions are not consistent. shape(cScaler, cLower, cUpper, cAdder) = ({cScaler.shape}, {cLower.shape}, {cUpper.shape}, {cAdder.shape})')
            # Note that c_lower, c_upper are both in scaled form
            self.c_lower  = (cLower+cAdder) * cScaler
            self.c_upper  = (cUpper+cAdder) * cScaler
            self.c_scaler = cScaler * 1.0
            self.c_adder  = cAdder * 1.0
        
        if oScaler is not None:
            if not (oScaler.shape == (1,)):
                raise ValueError(f'Objective scaler dimensions must be (1,) for single objective optimization. shape(oScaler) = {oScaler.shape}')
        self.o_scaler = oScaler * 1.0 if oScaler is not None else 1.0

        if oAdder is not None:
            if not (oAdder.shape == (1,)):
                raise ValueError(f'Objective adder dimensions must be (1,) for single objective optimization. shape(oAdder) = {oAdder.shape}')
        self.o_adder = oAdder * 1.0 if oAdder is not None else 0.0
        
        self.model_evals = 0                    # number of model evaluations
        self.deriv_evals = 0                    # number of derivative evaluations
        self.fail1 = False                      # failure of functions
        self.fail2 = False                      # failure of functions or derivatives
        self.warm_x       = self.x0 - 1.      # (x0 - 1.) to keep it different from initial dv values
        self.warm_x_deriv = self.x0 - 2.      # (x0 - 2.) to keep it different from initial dv and warm_x values
        self.declared_variables = ['dv', 'obj', 'grad']
        if self.nc > 0:
            self.constrained = True
            self.declared_variables += ['con', 'jac']

        self.SURF_mode = False # True if using SURF, False if using RS/FS
        
        # Run checks for the first model evaluation
        sim.update_design_variables(self.x0/self.x_scaler - self.x_adder)
        fun_dict = sim.compute_optimization_functions()
        f_us = fun_dict['f']
        c_us = fun_dict['c']
        
        if f_us is not None:
            if self.o_scaler is None:
                raise ValueError('Objective scaler is None but objective function returned a value. Please provide a valid scaler.')     
            if self.o_adder is None:
                raise ValueError('Objective adder is None but objective function returned a value. Please provide a valid adder.')
            if f_us.shape != (1,):
                raise ValueError(f'Only single objective optimization is supported but returned multiple objectives. shape(objective) = {f_us.shape}')
            if np.isnan(np.sum(f_us)) or np.isinf(np.sum(f_us)):
                raise Exception('Objective returned NAN/INF for the first run. Please check the model setup.')
            self.f_s = (f_us+self.o_adder) * self.o_scaler
        else:
            if self.o_scaler is not None:
                raise ValueError('Objective scaler is not None, but objective function returned None. Please check the model setup.')
            if self.o_adder is not None:
                raise ValueError('Objective adder is not None, but objective function returned None. Please check the model setup.')
            
        if c_us is not None:
            if self.c_scaler is None:
                raise ValueError('Constraint scaler is None but constraint function returned a value. Please provide a valid scaler.')
            if self.c_adder is None:
                raise ValueError('Constraint adder is None but constraint function returned a value. Please provide a valid adder.')
            if c_us.shape != (self.nc,):
                raise ValueError(f'Constraint function returned shape ({c_us.shape},) but ({self.nc},) was expected corresponding to provided cScaler, cLower, cUpper, and cAdder.')
            if np.isnan(np.sum(c_us)) or np.isinf(np.sum(c_us)):
                raise Exception('Constraints contain NAN/INF for the first run. Please check the model setup.')
            self.c_s = (c_us+self.c_adder) * self.c_scaler
        else:
            if self.c_scaler is not None:
                raise ValueError('Constraint scaler is not None, but constraint function returned None. Please check the model setup.')
            if self.c_adder is not None:
                raise ValueError('Constraint adder is not None, but constraint function returned None. Please check the model setup.')
            self.c_s = None
            
        self.model_evals += 1
        self.warm_x[:] = self.x0
            
        # Run checks for the first derivative evaluation
        deriv_dict = sim.compute_optimization_derivatives()
        g_us = deriv_dict['df']
        j_us = deriv_dict['dc']

        if g_us is not None:
            if self.o_scaler is None:
                raise ValueError('Objective scaler is None but objective gradient returned a value. Please provide a valid scaler.')
            if g_us.shape != (1, self.nx):
                raise ValueError(f'Objective gradient must be a row vector of shape (1, {self.nx}) but computed gradient has shape {g_us.shape}.')
            if np.isnan(np.sum(g_us)) or np.isinf(np.sum(g_us)):
                raise Exception('Objective gradient contains NAN/INF for the first run. Please check the model setup.')
            self.g_s = g_us[0] * self.o_scaler / self.x_scaler
        else:
            if self.o_scaler is not None:
                raise ValueError('Objective scaler is not None, but objective gradient returned None. Please check the model setup.')
            
        if j_us is not None:
            if self.c_scaler is None:
                raise ValueError('Constraint scaler is None but constraint Jacobian returned a value. Please provide a valid scaler.')
            if j_us.shape != (self.nc, self.nx):
                raise ValueError(f'Constraint Jacobian must be a matrix of shape ({self.nc}, {self.nx}) but computed Jacobian has shape {j_us.shape}.')
            if np.isnan(np.sum(j_us)) or np.isinf(np.sum(j_us)):
                raise Exception('Constraint Jacobian contains NAN/INF for the first run. Please check the model setup.')
            self.j_s = j_us * np.outer(self.c_scaler, 1./self.x_scaler)
        else:
            if self.c_scaler is not None:
                raise ValueError('Constraint scaler is not None, but constraint Jacobian returned None. Please check the model setup.')
            self.j_s = None

        self.deriv_evals += 1
        self.warm_x_deriv[:] = self.x0

    def setup_derivatives(self, ):
        pass

    def raise_issues_with_user_setup(self, ):
        pass
    
# Try except to get the failures from the simulator, but not for the first run
# If the first run fails, raise an exception
# If the subsequent runs fail, set the fail flag to True
# check_failure, guess_dict, tol_dict, force_rerun are not used in the current implementation
    
    def check_if_warm_and_run_model(self, x, guess_dict=None, tol_dict=None, 
                                    force_rerun=False, check_failure=False):
        '''
        Input x is the scaled design variable vector.
        '''

        sim = self.options['simulator']
        if not self.SURF_mode:      # for pure RS/FS 
            if (not np.array_equal(self.warm_x, x)) or force_rerun:
                sim.update_design_variables(x/self.x_scaler - self.x_adder)
                try:
                    fun_dict = sim.compute_optimization_functions()
                    f_us = fun_dict['f']
                    c_us = fun_dict['c']
                    self.fail1 = False
                    if f_us is not None:
                        self.f_s = (f_us+self.o_adder) * self.o_scaler
                        if np.isnan(np.sum(f_us)) or np.isinf(np.sum(f_us)):
                            raise Exception('Objective returned NAN/INF. Please check the model.')
                            
                    if c_us is not None:
                        self.c_s = (c_us+self.c_adder) * self.c_scaler
                        if np.isnan(np.sum(c_us)) or np.isinf(np.sum(c_us)):
                            raise Exception('Constraints contain NAN/INF. Please check the model.')

                except:
                    self.fail1 = True
                    if not check_failure:
                        raise Exception('Model evaluation failed. Please check the model.')
                    
                self.model_evals += 1
                self.warm_x[:] = x

            return
                
    def check_if_warm_and_compute_derivatives(self, x, guess_dict=None, tol_dict=None, 
                                              force_rerun=False, check_failure=False):
        sim = self.options['simulator']
        if not self.SURF_mode:      # for pure RS/FS                     
            if not np.array_equal(self.warm_x_deriv, x) or force_rerun:
                sim.update_design_variables(x/self.x_scaler - self.x_adder)
                try:
                    deriv_dict = sim.compute_optimization_derivatives()
                    f_us = deriv_dict['f']
                    c_us = deriv_dict['c']
                    g_us = deriv_dict['df']
                    j_us = deriv_dict['dc']
                    self.fail2 = False

                    if f_us is not None:
                        self.f_s = (f_us+self.o_adder) * self.o_scaler
                        if np.isnan(np.sum(f_us)) or np.isinf(np.sum(f_us)):
                            raise Exception('Objective returned NAN/INF. Please check the model.')
                        
                    if c_us is not None:
                        self.c_s = (c_us+self.c_adder) * self.c_scaler
                        if np.isnan(np.sum(c_us)) or np.isinf(np.sum(c_us)):
                            raise Exception('Constraints contain NAN/INF. Please check the model.')

                    if g_us is not None:
                        self.g_s = g_us[0] * self.o_scaler / self.x_scaler # g_us is a 2d matrix of gradients for each objective
                        if np.isnan(np.sum(g_us)) or np.isinf(np.sum(g_us)):
                            raise Exception('Objective gradient contains NAN/INF. Please check the model.')
                    
                    if j_us is not None:
                        self.j_s = j_us * np.outer(self.c_scaler, 1./self.x_scaler)
                        if np.isnan(np.sum(j_us)) or np.isinf(np.sum(j_us)):
                            raise Exception('Constraint Jacobian contains NAN/INF. Please check the model.')
                        
                except:
                    self.fail2 = True
                    if not check_failure:
                        raise Exception('Derivative evaluation failed. Please check the model.')
                        
                self.model_evals += 1
                self.warm_x[:] = x
                self.deriv_evals += 1
                self.warm_x_deriv[:] = x
                
            return

    def _setup_bounds(self): # x and c bounds don't include states y and residuals R for SURF
        pass
    def _setup_bounds(self): # x and c bounds don't include states y and residuals R for SURF
        pass

    def _setup_scalers(self):
        pass

    # TODO: Add decorators for checking if x is warm and for updating dvs
    @record(['x'], ['obj'])
    @hot_start(['x'], ['obj'])
    def _compute_objective(self, x, guess_dict=None, tol_dict=None, 
                           force_rerun=False, check_failure=False):
        # print('Computing objective >>>>>>>>>>')
        self.check_if_warm_and_run_model(x, guess_dict, tol_dict, 
                                         force_rerun, check_failure)
        # print('---------Computed objective---------')
        return self._get_objective()
        # return failure_flag, sim.objective()

    @record(['x'], ['grad'])
    @hot_start(['x'], ['grad'])
    def _compute_objective_gradient(self, x, guess_dict=None, tol_dict=None, 
                                    force_rerun=False, check_failure=False):
        # print('Computing gradient >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x, guess_dict, tol_dict, 
                                                   force_rerun, check_failure)
        # print('---------Computed gradient---------')
        return self._get_objective_gradient()

    @record(['x'], ['con'])
    @hot_start(['x'], ['con'])
    def _compute_constraints(self, x, guess_dict=None, tol_dict=None, 
                             force_rerun=False, check_failure=False):
        # print('Computing constraints >>>>>>>>>>')
        self.check_if_warm_and_run_model(x, guess_dict, tol_dict, 
                                         force_rerun, check_failure)
        # print('---------Computed constraints---------')
        return self._get_constraints()

    @record(['x'], ['jac'])
    @hot_start(['x'], ['jac'])
    def _compute_constraint_jacobian(self, x, guess_dict=None, tol_dict=None, 
                                     force_rerun=False, check_failure=False):
        # print('Computing Jacobian >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x, guess_dict, tol_dict, 
                                                   force_rerun, check_failure)
        # print('---------Computed Jacobian---------')
        return self._get_constraint_jacobian()

    @record(['x'], ['failure', 'obj', 'con', 'grad', 'jac'])
    @hot_start(['x'], ['failure', 'obj', 'con', 'grad', 'jac'])
    def _compute_all(self, x, force_rerun=False, check_failure=False):                              # only for SNOPTC, (NOT meant for SURF)
        # print('Computing all at once >>>>>>>>>>')
        self.check_if_warm_and_compute_derivatives(x, force_rerun=force_rerun, check_failure=check_failure)
        # print('---------Computed all at once---------')
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