from array_manager.api import VectorComponentsDict, MatrixComponentsDict, Vector, Matrix, BlockMatrix
from array_manager.api import DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix

from modopt.utils.options_dictionary import OptionsDictionary
from modopt.utils.general_utils import pad_name

import numpy as np
import warnings

from copy import deepcopy


class Problem(object):
    '''
    Base class for defining optimization problems in modOpt.

    Attributes
    ----------
    problem_name : str
        Problem name assigned by the user.
    x0 : np.ndarray
        Initial guess (scaled) for design variables.
    x : array_manager.Vector
        Current iterate (unscaled) for the design variables.
    nx : int
        Number of design variables or optimization variables.
    nc : int
        Number of constraints in the optimization problem.
    options : modopt.OptionsDictionary
        Problem-specific options declared by the user in addition
        to the global problem options 'jac_format' and 'hess_format'.
    obj : dict
        Dictionary with objective names as keys and current (unscaled) objective
        function values as values.
        Note that only one objective is supported by modOpt currently.
    obj_scaler : dict
        Dictionary with objective names as keys and objective.
        scalers as values. Default value for objective scalers is 1.0. 
    lag : dict
        Dictionary with the objective name as key and current Lagrangian
        function value as value.
    constrained: bool
        True if the problem has constraints. False if unconstrained.
    declared_variables: list
        List of problem variables declared by the user.
        It can at most be ['dv', 'obj', 'grad', 'con', 'jac', 'jvp', 'vjp', 
        'obj_hess', 'obj_hvp','lag_grad', 'lag_hess', 'lag_hvp']

    x_lower : np.ndarray or NoneType
        Vector of (scaled) lower bounds for the design variables.
        x_lower[k] = -np.inf if the variable at x[k] has no lower bound.
        x_lower = None if no variables have lower bounds.
    x_upper : np.ndarray or NoneType
        Vector of (scaled) upper bounds for the design variables.
        x_upper[k] = np.inf if the variable at x[k] has no upper bound.
        x_upper = None if no variables have upper bounds.
    c_lower : np.ndarray or NoneType
        Vector of (scaled) lower bounds for the constraints.
        c_lower[k] = -np.inf if the constraint at c[k] has no lower bound.
        c_lower = None if unconstrained or no constraints have lower bounds.
    c_upper : np.ndarray or NoneType
        Vector of (scaled) upper bounds for the constraints.
        c_upper[k] = np.inf if the constraint at c[k] has no upper bound.
        c_upper = None if unconstrained or no constraints have upper bounds.

    x_scaler : np.ndarray or NoneType
        Vector of scalers for the design variables.
        x_scaler[k] = 1.0 by default.
    c_scaler : np.ndarray or NoneType
        Vector of scalers for the constraints.
        c_scaler = None if unconstrained.
        c_scaler[k] = 1.0 by default.

    design_variables_dict : arraymanager.VectorComponentsDict
        Dictionary containing (unscaled) design variable vector metadata.
    constraints_dict : arraymanager.VectorComponentsDict
        Dictionary containing (unscaled) constraint vector metadata.

    pC_px_dict : arraymanager.MatrixComponentsDict
        Dictionary containing (unscaled) constraint Jacobian matrix metadata.
    p2F_pxx_dict : arraymanager.MatrixComponentsDict
        Dictionary containing (unscaled) objective Hessian matrix metadata.
    p2L_pxx_dict : arraymanager.MatrixComponentsDict
        Dictionary containing Lagrangian Hessian matrix metadata.

    pF_px : array_manager.Vector
        Abstract vector containing (unscaled) objective gradients.
    pL_px : array_manager.Vector
        Abstract vector containing Lagrangian gradients.
    
    con : array_manager.Vector
        Abstract vector containing (unscaled) constraints.
    jvp : array_manager.Vector
        Abstract vector containing constraint Jacobian-vector products (JVPs).
    vjp : array_manager.Vector
        Abstract vector containing constraint vector-Jacobian products (VJPs).
    obj_hvp : array_manager.Vector
        Abstract vector containing objective Hessian-vector products (HVPs).
    lag_hvp : array_manager.Vector
        Abstract vector containing Lagrangian Hessian-vector products (HVPs).

    pC_px : arraymanager.Matrix
        Abstract matrix containing (unscaled) constraint Jacobian components.
    p2F_pxx : arraymanager.Matrix
        Abstract matrix containing (unscaled) objective Hessian components.
    p2L_pxx : arraymanager.Matrix
        Abstract matrix containing Lagrangian Hessian components.

    jac : (array_manager.DenseMatrix, array_manager.COOMatrix, array_manager.CSRMatrix, array_manager.CSCMatrix)
        Standard array_manager matrix object in the format 
        self.options['jac_format'] provided by the user.
        This object provides standard numpy dense or scipy sparse matrices.
        The output format for matrices is useful to meet the requirements for the chosen optimizer.
    obj_hess : (array_manager.DenseMatrix, array_manager.COOMatrix, array_manager.CSRMatrix, array_manager.CSCMatrix)
        Standard array_manager matrix object in the format 
        self.options['hess_format'] provided by the user. 
        This object provides standard numpy dense or scipy sparse (unscaled) objective Hessian matrices.
        The output format for matrices is useful to meet the requirements for the chosen optimizer.
    lag_hess : (array_manager.DenseMatrix, array_manager.COOMatrix, array_manager.CSRMatrix, array_manager.CSCMatrix)
        Standard array_manager matrix object in the format 
        self.options['hess_format'] provided by the user. 
        This object provides standard numpy dense or scipy sparse matrices.
        The output format for matrices is useful to meet the requirements for the chosen optimizer.
    '''
    def __init__(self, **kwargs):
        '''
        Initialize the Problem() object.
        Calls user-specified initialize() method, 
        and _setup() method.
        '''

        self.options = OptionsDictionary()

        self.problem_name = 'unnamed_problem'
        self.options.declare('jac_format',
                             default='dense',
                             values=('dense', 'coo', 'csr', 'csc'))
        self.options.declare('hess_format',
                             default='dense',
                             values=('dense', 'coo', 'csr', 'csc'))

        self.x0 = None
        self.nx = 0
        self.nc = 0
        # TODO: Fix this
        self.obj = {}
        self.obj_scaler = {}
        self.lag = {}
        self.constrained = False
        self.declared_variables = []

        ###############################
        # Only for the SURF algorithm #
        ###############################
        self.y0 = None
        self.ny = 0
        self.nr = 0
        self.implicit_constrained = False
        ###############################

        self.x_lower = None
        self.x_upper = None
        self.x_scaler = None
        self.c_lower = None
        self.c_upper = None
        self.c_scaler = None

        self.initialize()
        self.options.update(kwargs)

        self.design_variables_dict = VectorComponentsDict()
        self.constraints_dict = VectorComponentsDict()

        ###############################
        # Only for the SURF algorithm #
        ###############################
        self.state_variables_dict = VectorComponentsDict()
        self.residuals_dict = VectorComponentsDict()
        ###############################

        self._setup()

    def _setup(self):
        '''
        Calls user-specified setup() method, then _setup_scalers(), _setup_bounds().
        Sets up vectors for pF_px, obj_hvp.
        Sets up vectors for pL_px, jvp, vjp, and lag_hvp, if there are 
        constraints declared in the setup().
        Sets up a MatrixComponentsDict() object for the objective Hessian.
        Sets up MatrixComponentsDict() objects for the constraint Jacobian 
        and Lagrangian Hessian, if there are constraints.
        
        Calls user-specified setup_derivatives().
        Sets up matrices for Jacobian (if there are constraints),
        objective or Lagrangian Hessian depending on which one is
        declared by the user.

        Finally delete any unnecessary attributes allocated but
        was not declared by the user.
        For example, vjp, jvp, pL_px, obj_hvp, lag_hvp, 
        p2F_pxx_dict, p2L_pxx_dict, etc.
        '''
        self.setup()

        self._setup_scalers()
        # CSDLProblem() overrides this method in Problem()
        self._setup_bounds()
        self._setup_vectors()

        # When problem is not defined as CSDLProblem() [Note: self.x0 is always scaled]
        if self.x0 is None:
            # Note: array_manager puts np.zeros as the initial guess if no initial guess is provided
            self.x0 = self.x.get_data() * self.x_scaler
    
        self._setup_jacobian_dict()
        self._setup_hessian_dict()

        self.setup_derivatives()
        self._setup_matrices()

        self.delete_unnecessary_attributes_allocated()
        self.raise_issues_with_user_setup()

    def __str__(self):
        """
        Print the details of the optimization problem.
        """
        name = self.problem_name
        obj  = self.obj
        obj_scaler = self.obj_scaler
        dvs  = self.x
        x_s  = self.x_scaler; x_l  = self.x_lower/x_s; x_u  = self.x_upper/x_s 
        if self.constrained:
            cons = self.con
            c_s  = self.c_scaler; c_l  = self.c_lower/c_s; c_u  = self.c_upper/c_s

        # output  = '\n\t'+'-'*100
        output  = f'\n\tProblem Overview:\n\t' + '-'*100
        output += f'\n\t' + pad_name('Problem name', 25) + f': {name}'
        
        # Print objective name
        # if len(obj) > 1:
        #     raise (f'More than one objective defined for the optimization problem: {", ".join(obj)}.')
        # output += f'\n\tObjective: {list(obj.keys())[0]}'
        output += f'\n\t' + pad_name('Objectives', 25) + f': '+', '.join(obj.keys())
        
        # Print design variables list with their dimensions
        dv_list = ''
        for dv_name, dv in dvs.dict_.items():
            dv_list += f'{dv_name} {dv.shape}, '
        dv_list = dv_list[:-2]
        output += f'\n\t' + pad_name('Design variables', 25) + f': {dv_list}'
        
        #  Print constraints and their dimensions
        if self.constrained:
            con_list = ''
            for con_name, con in cons.dict_.items():
                con_list += f'{con_name} {con.shape}, '
            con_list = con_list[:-2]
            output += f'\n\t' + pad_name('Constraints', 25) + f': {con_list}'

        output += '\n\t' + '-'*100
        
        output += f'\n\n\tProblem Data (UNSCALED):\n\t' + '-'*100
        
        # Print objective data
        output += f'\n\tObjectives:\n'
        header = "\t%-5s | %-10s | %-13s | %-13s " % ('Index', 'Name', 'Scaler', 'Value')
        output += header
        obj_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {value:<+.6e}"
        for i, obj_name in enumerate(obj.keys()):
            obj_value   = obj[obj_name]
            obj_s       = obj_scaler[obj_name]
            obj_name    = obj_name[:10] if (len(obj_name)>10) else obj_name
            output     += obj_template.format(idx=i, name=obj_name, scaler=obj_s, value=obj_value)
        
        # Print design variable data
        output += f'\n\n\tDesign Variables:\n'
        header = "\t%-5s | %-10s | %-13s | %-13s | %-13s | %-13s " % ('Index', 'Name', 'Scaler', 'Lower Limit', 'Value', 'Upper Limit')
        output += header
        dv_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {lower:<+.6e} | {value:<+.6e} | {upper:<+.6e}"

        idx = 0
        for dv_name, dv in dvs.dict_.items():
            for i, x in enumerate(dv.flatten()):
                dv_name  = dv_name[:10] if (len(dv_name)>10) else dv_name
                l = -1.e99 if x_l[idx] == -np.inf else x_l[idx]
                u = +1.e99 if x_u[idx] == +np.inf else x_u[idx]
                output += dv_template.format(idx=idx, name=dv_name+f'[{i}]', scaler=x_s[idx], lower=l, value=x, upper=u)
                idx += 1

        # Print constraint data
        if self.constrained:
            output += f'\n\n\tConstraints:\n'
            header = "\t%-5s | %-10s | %-13s | %-13s | %-13s | %-13s | %-13s " % ('Index', 'Name', 'Scaler','Lower Limit', 'Value', 'Upper Limit', 'Lag. mult.')
            output += header

            idx = 0
            lag_declared = any(x in self.declared_variables for x in ['lag_grad', 'lag_hess', 'lag_hvp'])
            if lag_declared: # print lagrange multipliers
                con_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {lower:<+.6e} | {value:<+.6e} | {upper:<+.6e} | {lag:<+.6e}"
                obj_s = list(obj_scaler.values())[0]
                z = self.lag_mult.get_data() * obj_s / c_s
            else:
                con_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {lower:<+.6e} | {value:<+.6e} | {upper:<+.6e} | "

            for con_name, con in cons.dict_.items():
                for i, c in enumerate(con.flatten()):
                    con_name  = con_name[:10] if (len(con_name)>10) else con_name
                    l = -1.e99 if c_l[idx] == -np.inf else c_l[idx]
                    u = +1.e99 if c_u[idx] == +np.inf else c_u[idx]
                    if lag_declared: # print lagrange multipliers
                        output += con_template.format(idx=idx, name=con_name+f'[{i}]', scaler=c_s[idx], lower=l, value=c, upper=u, lag=z[idx])
                    else:
                        output += con_template.format(idx=idx, name=con_name+f'[{i}]', scaler=c_s[idx], lower=l, value=c, upper=u)
                    idx += 1

        output += '\n\t' + '-'*100 + '\n'
            
        return output

    def initialize(self):
        '''
        User-defined method.
        Set problem name and any problem-specific options.
        '''
        pass

    def setup(self):
        '''
        User-defined method.
        Call add_design_variables(), add_objective(), and add_constraints() inside.
        '''
        pass

    def setup_derivatives(self, ):
        '''
        User-defined method.
        Call declare_objective_gradient(), declare_objective_hessian(), declare_objective_hvp(),
        declare_constraint_jacobian(), declare_constraint_jvp(), declare_constraint_vjp(), 
        declare_lagrangian_gradient(), declare_lagrangian_hessian(), and declare_lagrangian_hvp() 
        inside.
        '''
        pass

    def _setup_scalers(self):
        '''
        Set x_scaler and c_scaler attributes using design_variables_dict and constraints_dict.
        c_scaler = None if there are no constraints.
        '''
        self.x_scaler = self.design_variables_dict.scaler
        # When unconstrained: self.constraints_dict.scaler = []
        if self.constrained:
            self.c_scaler = self.constraints_dict.scaler

        # Use this if componentwise dv scalers or constraint scalers are needed
        # self.x_scaler_abstract = Vector(self.design_variables_dict)
        # self.x_scaler_abstract.allocate(data=self.x_scaler, setup_views=True)
        # if self.constrained:
        #     self.c_scaler_abstract = Vector(self.constraints_dict)
        #     self.c_scaler_abstract.allocate(data=self.c_scaler, setup_views=True)

    def _setup_bounds(self):
        '''
        Compute scaled bounds x_lower, x_upper, c_lower, 
        and c_upper for the optimizer.
        THis method is overridden in CSDLProblem(), OpenMDAOProblem(), CUTEstProblem().
        Note that c_upper = None and c_lower = None if there are no constraints.
        '''
        self.x_lower = self.design_variables_dict.lower * self.x_scaler
        self.x_upper = self.design_variables_dict.upper * self.x_scaler
        # When unconstrained: self.constraints_dict.lower = [], self.constraints_dict.lower = []
        if self.constrained:
            self.c_lower = self.constraints_dict.lower * self.c_scaler
            self.c_upper = self.constraints_dict.upper * self.c_scaler   

    def _setup_vectors(self):
        '''
        Set up array_manager abstract vectors for design variables 'x',
        objective gradients 'pF_px', objective HVP 'obj_hvp'.
        If constrained, also set up abstract vectors for constraints 'con',
        constraint JVP 'jvp', constraint VJP 'vjp', 
        Lagrangian gradients 'pL_px', Lagrangian HVP 'lag_hvp'.
        '''
        self.x = Vector(self.design_variables_dict)
        self.x.allocate(setup_views=True)
        self.pF_px = Vector(self.design_variables_dict)
        self.pF_px.allocate(data=np.zeros((self.nx, )),
                            setup_views=True)
        self.obj_hvp = Vector(self.design_variables_dict)
        self.obj_hvp.allocate(data=np.zeros((self.nx, )),
                              setup_views=True)
        self.vec_hvp = Vector(self.design_variables_dict)
        self.vec_hvp.allocate(data=np.zeros((self.nx, )),
                              setup_views=True)
        
        if self.constrained:
            self.con = Vector(self.constraints_dict)
            self.con.allocate(setup_views=True)

            self.lag_mult = Vector(self.constraints_dict)
            self.lag_mult.allocate(data=np.zeros((self.nc, )),
                                   setup_views=True)

            self.jvp = Vector(self.constraints_dict)
            self.jvp.allocate(data=np.zeros((self.nc, )),
                              setup_views=True)
            self.vec_jvp = Vector(self.design_variables_dict)
            self.vec_jvp.allocate(data=np.zeros((self.nx, )),
                                  setup_views=True)
            
            self.vjp = Vector(self.design_variables_dict)
            self.vjp.allocate(data=np.zeros((self.nx, )),
                              setup_views=True)
            self.vec_vjp = Vector(self.constraints_dict)
            self.vec_vjp.allocate(data=np.zeros((self.nc, )),
                                  setup_views=True)
            
            self.pL_px = Vector(self.design_variables_dict)
            self.pL_px.allocate(data=np.zeros((self.nx, )),
                                setup_views=True)
            self.lag_hvp = Vector(self.design_variables_dict)
            self.lag_hvp.allocate(data=np.zeros((self.nx, )),
                                  setup_views=True)

        ###############################
        # Only for the SURF algorithm #
        ###############################
        if self.implicit_constrained:
            self.y = Vector(self.state_variables_dict)
            self.y.allocate(setup_views=True)

            self.pF_py = Vector(self.design_variables_dict)
            self.pF_py.allocate(data=np.zeros((self.ny, )),
                                setup_views=True)

            self.residuals = Vector(self.residuals_dict)
            self.residuals.allocate(setup_views=True)
        ###############################

        # self.constraint_duals = Vector(self.constraints_dict)
        # self.residual_duals = Vector(self.residuals_dict) 

    def _setup_jacobian_dict(self):
        '''
        Set up array_manager MatrixComponentDict for constraint Jacobians,
        if the problem is constrained.
        '''
        if self.constrained:
            self.pC_px_dict = MatrixComponentsDict(
                self.constraints_dict, self.design_variables_dict)
            
        ###############################
        # Only for the SURF algorithm #
        ###############################
            if self.implicit_constrained:
                self.pC_py_dict = MatrixComponentsDict(
                    self.constraints_dict, self.state_variables_dict)       
        
        if self.implicit_constrained:
            self.pR_px_dict = MatrixComponentsDict(
                self.residuals_dict, self.design_variables_dict)
            self.pR_py_dict = MatrixComponentsDict(
                self.residuals_dict, self.state_variables_dict)
        ###############################

    def _setup_hessian_dict(self):
        '''
        Set up array_manager MatrixComponentDict for objective Hessians.
        Also set up MatrixComponentDict for Lagrangian Hessians,
        if the problem is constrained.       
        '''
        self.p2F_pxx_dict = MatrixComponentsDict(
            self.design_variables_dict, self.design_variables_dict)

        if self.constrained:
            self.p2L_pxx_dict = MatrixComponentsDict(
                self.design_variables_dict, self.design_variables_dict)
        
        ###############################
        # Only for the SURF algorithm #
        ###############################
            if self.implicit_constrained:
                self.p2L_pxy_dict = MatrixComponentsDict(
                    self.design_variables_dict, self.state_variables_dict)
                self.p2L_pyy_dict = MatrixComponentsDict(
                    self.state_variables_dict, self.state_variables_dict)

        elif self.implicit_constrained:
            self.p2L_pxx_dict = MatrixComponentsDict(
                self.design_variables_dict, self.design_variables_dict)
            self.p2L_pxy_dict = MatrixComponentsDict(
                self.design_variables_dict, self.state_variables_dict)
            self.p2L_pyy_dict = MatrixComponentsDict(
                self.state_variables_dict, self.state_variables_dict)
        ###############################

    def _setup_matrices(self):
        '''
        Set up array_manager native Matrix and standard Matrix for 
        objective/Lagrangian Hessian depending on whether they are declared.
        Also set up native Matrix and standard Matrix for constraint Jacobian,
        if the problem is constrained.   
        '''
        if 'obj_hess' in self.declared_variables:
            self.p2F_pxx = Matrix(self.p2F_pxx_dict, setup_views=True)
            self.p2F_pxx.allocate()
            if self.options['hess_format'] == 'dense':
                self.obj_hess = DenseMatrix(self.p2F_pxx)
            elif self.options['hess_format'] == 'coo':
                self.obj_hess = COOMatrix(self.p2F_pxx)
            elif self.options['hess_format'] == 'csr':
                self.obj_hess = CSRMatrix(self.p2F_pxx)
            else:
                self.obj_hess = CSCMatrix(self.p2F_pxx)

        if self.constrained:
            self.pC_px = Matrix(self.pC_px_dict, setup_views=True)
            self.pC_px.allocate()
            # TODO: add standard matrices for all jac and hess
            if self.options['jac_format'] == 'dense':
                self.jac = DenseMatrix(self.pC_px)
            elif self.options['jac_format'] == 'coo':
                self.jac = COOMatrix(self.pC_px)
            elif self.options['jac_format'] == 'csr':
                self.jac = CSRMatrix(self.pC_px)
            else:
                self.jac = CSCMatrix(self.pC_px)

            if 'lag_hess' in self.declared_variables:    
                self.p2L_pxx = Matrix(self.p2L_pxx_dict, setup_views=True)
                self.p2L_pxx.allocate()
                if self.options['hess_format'] == 'dense':
                    self.lag_hess = DenseMatrix(self.p2L_pxx)
                elif self.options['hess_format'] == 'coo':
                    self.lag_hess = COOMatrix(self.p2L_pxx)
                elif self.options['hess_format'] == 'csr':
                    self.lag_hess = CSRMatrix(self.p2L_pxx)
                else:
                    self.lag_hess = CSCMatrix(self.p2L_pxx)

        ###############################
        # Only for the SURF algorithm #
        ###############################        

                if self.implicit_constrained:
                    self.p2L_pxy = Matrix(self.p2L_pxy_dict, setup_views=True)
                    self.p2L_pxy.allocate()
                    self.p2L_pyy = Matrix(self.p2L_pyy_dict, setup_views=True)
                    self.p2L_pyy.allocate()

            if self.implicit_constrained:
                self.pC_py = Matrix(self.pC_py_dict, setup_views=True)
                self.pC_py.allocate()
                self.pR_px = Matrix(self.pR_px_dict, setup_views=True)
                self.pR_px.allocate()
                self.pR_py = Matrix(self.pR_py_dict, setup_views=True)
                self.pR_py.allocate()

        elif self.implicit_constrained:
            self.pR_px = Matrix(self.pR_px_dict, setup_views=True)
            self.pR_px.allocate()
            self.pR_py = Matrix(self.pR_py_dict, setup_views=True)
            self.pR_py.allocate()
            if 'lag_hess' in self.declared_variables:
                self.p2L_pxx = Matrix(self.p2L_pxx_dict, setup_views=True)
                self.p2L_pxx.allocate()    
                self.p2L_pxy = Matrix(self.p2L_pxy_dict, setup_views=True)
                self.p2L_pxy.allocate()
                self.p2L_pyy = Matrix(self.p2L_pyy_dict, setup_views=True)
                self.p2L_pyy.allocate()
        ###############################
            
    def delete_unnecessary_attributes_allocated(self, ):
        '''
        After checking if 'grad', 'jac', 'obj_hess', 'obj_hvp', 'jvp', 'vjp', 'lag_grad', 
        'lag_hess', or 'lag_hvp' are in self.declared_variables,
        delete attributes self.pF_px, self.pC_px_dict, self.pC_px, self.jvp, self.vjp, self.obj_hess, self.obj_hvp, self.lag_grad, self.lag_hess, self.lag_hvp,
        and associated VectorComponentsDict() objects if not declared by the user.
        '''
        # NOTE: This memory deallocation is not implemented for SURF  

        if 'grad' not in self.declared_variables: # not needed for gradient-free optimization
            print('Deleting self.pFpx ...')
            del self.pF_px
        if 'obj_hess' not in self.declared_variables:
            del self.p2F_pxx_dict
        if 'obj_hvp' not in self.declared_variables:
            del self.obj_hvp
        if ('obj_hvp' not in self.declared_variables) and ('lag_hvp' not in self.declared_variables):
            del self.vec_hvp

        if self.constrained:
            if 'jac' not in self.declared_variables: # not needed for gradient-free optimization
                del self.pC_px_dict, self.pC_px
                print('Deleting self.pCpx, pCpx_dict ...')
            if 'jvp' not in self.declared_variables:
                del self.jvp, self.vec_jvp
            if 'vjp' not in self.declared_variables:
                del self.vjp, self.vec_vjp
            if 'lag_grad' not in self.declared_variables:
                del self.pL_px
            if 'lag_hess' not in self.declared_variables:
                del self.p2L_pxx_dict
            if 'lag_hvp' not in self.declared_variables:
                del self.lag_hvp
            if all(x not in self.declared_variables for x in ['lag_grad', 'lag_hess', 'lag_hvp']):
                del self.lag_mult
        else:
            del self.constraints_dict

    def raise_issues_with_user_setup(self, ):
        '''
        Raise errors or warnings associated with declarations made by the user in
        setup() or setup_derivatives().
        Overridden when using interfaced modeling frameworks like CSDL or OpenMDAO.
        '''
        if 'dv' not in self.declared_variables:
            raise Exception("No design variables are declared.")
        if 'obj' in self.declared_variables:
            if self.compute_objective.__func__ == Problem.compute_objective:
                raise Exception("Objective is declared but compute_objective() method is not implemented.")
        else:
            if 'con' not in self.declared_variables:
                raise Exception("No objective or constraints are declared.")
            warnings.warn("No objective is declared. Running a feasibility problem.")
            self.add_objective('dummy_obj')
            self.obj['dummy_obj'] = 0. # Default value 1. is replaced with 0. for feasibility problems

            # Add back pF_px only for a gradient-based feasibility problem since 
            # it was deleted in delete_unnecessary_attributes_allocated()
            if 'jac' in self.declared_variables: # checking if gradients are declared for constraints
                self.pF_px = Vector(self.design_variables_dict)
                self.pF_px.allocate(data=np.zeros((self.nx, )), setup_views=False)
        if 'con' in self.declared_variables:
            if self.compute_constraints.__func__ == Problem.compute_constraints:
                raise Exception("Constraints are declared but compute_constraints() method is not implemented.")
        if 'grad' in self.declared_variables:
            if self.compute_objective_gradient.__func__ == Problem.compute_objective_gradient:
                raise Exception("Objective gradient is declared but compute_objective_gradient() method is not implemented."
                                "If declared derivatives are constant, define an empty compute_objective_gradient() with 'pass'."
                                "If declared derivatives are not available, define a compute_objective_gradient() method"
                                "that calls self.use_finite_differencing('objective_gradient', step=1.e-6)."
                                "If using a gradient-free optimizer, do not declare objective gradient.")
        if 'obj_hess' in self.declared_variables:
            if self.compute_objective_hessian.__func__ == Problem.compute_objective_hessian:
                raise Exception("Objective Hessian is declared but compute_objective_hessian() method is not implemented."
                                "If declared derivatives are constant, define an empty compute_objective_hessian() with 'pass'."
                                "If declared derivatives are not available, define a compute_objective_hessian() method"
                                "that calls self.use_finite_differencing('objective_hessian', step=1.e-6)."
                                "If using a gradient-free optimizer, do not declare objective Hessian.")
        if 'obj_hvp' in self.declared_variables:
            if self.compute_objective_hvp.__func__ == Problem.compute_objective_hvp:
                raise Exception("Objective HVP is declared but compute_objective_hvp() method is not implemented.")
            
        if self.constrained:
            if 'jac' in self.declared_variables:
                if self.compute_constraint_jacobian.__func__ == Problem.compute_constraint_jacobian:
                    raise Exception("Constraint Jacobian is declared but compute_constraint_jacobian() method is not implemented."
                                    "If declared derivatives are constant, define an empty compute_constraint_jacobian() with 'pass'."
                                    "If declared derivatives are not available, define a compute_constraint_jacobian() method"
                                    "that calls self.use_finite_differencing('constraint_jacobian', step=1.e-6)."
                                    "If using a gradient-free optimizer, do not declare constraint Jacobian.")
            if 'jvp' in self.declared_variables:
                if self.compute_constraint_jvp.__func__ == Problem.compute_constraint_jvp:
                    raise Exception("Constraint JVP is declared but compute_constraint_jvp() method is not implemented.")
            if 'vjp' in self.declared_variables:
                if self.compute_constraint_vjp.__func__ == Problem.compute_constraint_vjp:
                    raise Exception("Constraint VJP is declared but compute_constraint_vjp() method is not implemented.")
            if 'lag_grad' in self.declared_variables:
                if self.compute_lagrangian_gradient.__func__ == Problem.compute_lagrangian_gradient:
                    raise Exception("Lagrangian gradient is declared but compute_lagrangian_gradient() method is not implemented.")
            if 'lag_hess' in self.declared_variables:
                if self.compute_lagrangian_hessian.__func__ == Problem.compute_lagrangian_hessian:
                    raise Exception("Lagrangian Hessian is declared but compute_lagrangian_hessian() method is not implemented.")
            if 'lag_hvp' in self.declared_variables:
                if self.compute_lagrangian_hvp.__func__ == Problem.compute_lagrangian_hvp:
                    raise Exception("Lagrangian HVP is declared but compute_lagrangian_hvp() method is not implemented.")

        if ('grad' not in self.declared_variables) and ('lag_grad' not in self.declared_variables):
            # Don't raise an error since gradient-free optimization is possible
            # Will raise errors if trying to access gradients later since pF_px was deleted
            warnings.warn("No objective/Lagrangian gradient is declared.")

        if self.constrained:
            if all(x not in self.declared_variables for x in ['jac', 'jvp', 'vjp', 'lag_grad']):
                # Don't raise an error since gradient-free optimization is possible
                warnings.warn("No constraint-related derivatives (jacobian, jvp, vjp, dL/dx) are declared.")

    def add_design_variables(self,
                             name=None,
                             shape=(1, ),
                             scaler=None,
                             lower=None,
                             upper=None,
                             equals=None,
                             vals=None):
        '''
        User calls this method within Problem.setup() method
        to add design variable vectors for the problem.

        Parameters
        ----------
        name : str
            Design variable name assigned by the user.
        shape : tuple, default=(1,)
            Design variable shape. (1,) by default.
        scaler : float or np.ndarray, optional
            Design variable scaling factor.
            It can be a single scaler for all variables in the vector,
            or an array of scalers with the same shape as the design variable.
        lower : float or np.ndarray, optional
            Design variable lower bound.
            It can be a float in which case the given lower bound applies to all variables
            in the design variable vector.
            An array of lower bounds with the same shape as the design variable is also
            acceptable.
        upper: float or np.ndarray, optional
            Design variable upper bound.
            It can be a float in which case the given upper bound applies to all variables
            in the design variable vector.
            An array of upper bounds with the same shape as the design variable is also
            acceptable.
        equals: float or np.ndarray, optional
            Employing this makes the design variable a fixed constant.
            This must be used only for debugging purposes.
        vals: float or np.ndarray, optional
            Initial values for the design variables.
            It can be a single value for all variables in the vector,
            or an array of initial values with the same shape as the design variable.
            If nothing is provided, 0. will be taken as the initial guess.
        '''
        # array_manager automatically sets vals = np.zeros(size) if vals is None
        # Autonaming index starts from x0; for entries inside: xi_j (i dv_index, j dv_sub_index)
        if name is None:
            raise ValueError('A name must be provided for adding design variables.')
            # name = 'x' + str(len(self.design_variables_dict))

        self.design_variables_dict[name] = dict(
            shape=shape,
            scaler=scaler,
            lower=lower,
            upper=upper,
            equals=equals,
            vals=vals,
        )

        if 'dv' not in self.declared_variables:
            self.declared_variables.append('dv')

        # Update the number of design variables
        self.nx += np.prod(shape)

    def add_objective(self, name='obj', scaler=1.0):
        '''
        User calls this method within Problem.setup() method
        to add an objective with a name and a scaler.
        It also adds a Lagrangian dict with the objective name
        as key and default value as 1.0.

        Parameters
        ----------
        name : str, default='obj'
            Objective name assigned by the user.
        scaler : float, default=1.
            Objective scaling factor.
        '''
        # Setting objective name and initializing it with key=name and value=1.
        if len(self.obj)>0:
            raise KeyError('Only one objective is allowed for a problem.')

        print(f'Setting objective name as "{name}".')
        self.obj[name] = 1.
        self.obj_scaler[name] = scaler
        self.lag[name] = 1.

        if 'obj' not in self.declared_variables:
            self.declared_variables.append('obj')

    def add_constraints(self,
                        name=None,
                        shape=(1, ),
                        scaler=None,
                        lower=None,
                        upper=None,
                        equals=None):
        '''
        User calls this method within Problem.setup() method
        to add constraints for the problem.

        Parameters
        ----------
        name : str
            Constraint name assigned by the user.
        shape : tuple, default=(1,)
            Constraint shape. (1,) by default.
        scaler : float or np.ndarray, optional
            Constraint scaling factor.
            It can be a single scaler for all constraints in the vector,
            or an array of scalers with the same shape as the constraint.
        lower : float or np.ndarray, optional
            Constraint lower bound.
            It can be a float in which case the given lower bound applies to all constraints
            in the constraint vector.
            An array of lower bounds with the same shape as the constraint is also
            acceptable.
        upper: float or np.ndarray, optional
            Constraint upper bound.
            It can be a float in which case the given upper bound applies to all constraints
            in the constraint vector.
            An array of upper bounds with the same shape as the constraint is also
            acceptable.
        equals: float or np.ndarray, optional
            Used for defining an equality constraint.
            It can be a float in which case the given constant applies to all constraints
            in the constraint vector.
            An array of floats with the same shape as the constraint is also acceptable.
            It is used when the right-hand side constants for the equality constraints
            are different.
        '''
        # Autonaming index starts from 0; for entries inside: ci_j (i con_index, j con_sub_index)
        if name is None:
            raise ValueError('A name must be provided for adding constraints.')
            # name = 'c' + str(len(self.constraints_dict))

        self.constraints_dict[name] = dict(
            shape=shape,
            scaler=scaler,
            lower=lower,
            upper=upper,
            equals=equals,
        )

        if not self.constrained:
            self.constrained = True
        
        if 'con' not in self.declared_variables:
            self.declared_variables.append('con')
        
        # Update the number of constraints
        self.nc += np.prod(shape)

    def declare_objective_gradient(self, wrt, vals=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare nonzero objective gradients.
        Gradients that are undeclared are assumed to be zeros.
        
        Parameters
        ----------
        wrt : str
            Variable w.r.t. which the objective gradient needs to be declared.
        vals : float or np.ndarray, optional
            Values for constant gradients. 
            Useful if the objective is independent of, 
            or linearly-dependent on the declared "wrt" design variables.
        '''
        if wrt not in self.design_variables_dict:
            raise KeyError(
                'Undeclared design variable {} with respect to which gradient of the objective is declared.'
                .format(wrt))
        
        if 'grad' not in self.declared_variables:
            self.declared_variables.append('grad')
        if vals is not None:
            self.pF_px[wrt] = vals

    def declare_lagrangian_gradient(self, wrt, vals=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare nonzero Lagrangian gradients.
        Gradients that are undeclared are assumed to be zeros.
        
        Parameters
        ----------
        wrt : str
            Variable w.r.t. which the Lagrangian gradient needs to be declared.
        vals : float or np.ndarray, optional
            Values for constant gradients. 
            Useful if the Lagrangian is only linearly-dependent 
            on the declared "wrt" design variables.
        '''
        if wrt not in self.design_variables_dict:
            raise KeyError(
                'Undeclared design variable {} with respect to which gradient of the Lagrangian is declared.'
                .format(wrt))
        
        if 'lag_grad' not in self.declared_variables:
            self.declared_variables.append('lag_grad')
        if vals is not None:
            self.pL_px[wrt] = vals

    # If not declared, jacobians and gradients are assumed zero. 
    # TODO: If declared with no values, method = cs, fd 
    # Currently, user must provide the partials if using Problem() class 
    # since the functions must be fairly simple in order to use only the Problem() class or 
    # must be coupled through advanced frameworks like OpenMDAO/csdl for complex models 
    # and the frameworks already implement cs/fd approximations.

    def declare_constraint_jacobian(self,
                                    of,
                                    wrt,
                                    shape=None,
                                    vals=None,
                                    rows=None,
                                    cols=None,
                                    ind_ptr=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare nonzero constraint Jacobians.
        Jacobian components that are undeclared are assumed to be zeros.
        If the Jacobian is provided later (in compute_constraint_jacobian() method) 
        in one of the sparse formats coo, csr, or csc, declare the sparsity 
        by calling this method with kwargs (rows, cols), (rows, ind_ptr), 
        or (cols, ind_ptr), respectively.

        Parameters
        ----------
        of : str
            Name of the constraint for which the Jacobian needs to be declared.
        wrt : str
            Name of the variable w.r.t. which the Jacobian needs to be declared.
        rows : np.ndarray, optional
            Row indices corresponding to vals. 
            Needs to be declared if the format for the declared Jacobian is coo or csr.
        cols : np.ndarray, optional
            Column indices corresponding to vals. 
            Needs to be declared if the format for the declared Jacobian is coo or csc.
        ind_ptr : np.ndarray, optional
            Index pointer array for compressed index formats.
            Needs to be declared if the format for the declared Jacobian is csr or csc.
        shape : tuple, optional
            Shape in which 'vals' are going to be provided later by the user 
            (for dense or sparse formats).
            Note that this is not the shape of the declared Jacobian.
        vals : float or np.ndarray, optional
            Values for the constraint Jacobian. Useful if the constraint is 
            independent of or linearly-dependent on the declared "wrt" design variables.
            'vals' are nonzero entries corresponding to 'rows' or 'cols' if the 
            declared Jacobian is in sparse format.
        '''
        if of not in self.constraints_dict:
            raise KeyError(f'Jacobian is declared for undeclared constraint {of}.')
        if wrt not in self.design_variables_dict:
            raise KeyError(f'Jacobian is declared with respect to undeclared design variable {wrt}')

        if 'jac' not in self.declared_variables:
            self.declared_variables.append('jac')
        
        self.pC_px_dict[of, wrt] = dict(
            vals=vals,
            rows=rows,
            cols=cols,
            ind_ptr=ind_ptr,
            vals_shape=shape,
        )

    def declare_constraint_jvp(self, of, vals=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare constraint Jacobian-vector product (JVP).

        Parameters
        ----------
        of : str
            Name of the constraint for which the JVP needs to be declared.
        vals : float or np.ndarray, optional
            Values for the constraint JVP. Useful if the "of" constraint is 
            only linearly-dependent on all of the design variables.
        '''
        if of not in self.constraints_dict:
            raise KeyError(f'JVP is declared for undeclared constraint {of}.')

        if 'jvp' not in self.declared_variables:
            self.declared_variables.append('jvp')
        if vals is not None:
            self.jvp[of] = vals

    def declare_constraint_vjp(self, wrt, vals=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare constraint vector-Jacobian product (VJP).

        Parameters
        ----------
        wrt : str
            Name of the variable w.r.t. which the VJP needs to be declared.
        vals : float or np.ndarray, optional
            Values for the constraint VJP. Useful if all the constraints are 
            independent of or linearly-dependent on the "wrt" design variables.
        '''
        if wrt not in self.design_variables_dict:
            raise KeyError(f'VJP is declared with respect to undeclared design variable {wrt}.')

        if 'vjp' not in self.declared_variables:
            self.declared_variables.append('vjp')
        if vals is not None:
            self.vjp[wrt] = vals

    def declare_objective_hessian(self,
                                  of,
                                  wrt,
                                  shape=None,
                                  vals=None,
                                  rows=None,
                                  cols=None,
                                  ind_ptr=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare nonzero objective Hessian components F_{xy} = d^2F/dydx.
        Hessian components that are undeclared are assumed to be zeros.
        If the Hessian component is provided later (in compute_objective_hessian() method) 
        in one of the sparse formats coo, csr, or csc,
        declare the Hessian sparsity by calling this method with kwargs (rows, cols), (rows, ind_ptr), 
        or (cols, ind_ptr), respectively.

        Parameters
        ----------
        of : str
            Name of the variable x in F_{xy}.
            Note that the declared Hessian is the derivative of 
            the gradient "dF/dx" with respect to y, hence the keyword "of".
        wrt : str
            Name of the variable y in F_{xy}.
            Note that the declared Hessian is the derivative of
            the gradient dF/dx with respect to "y", hence the keyword "wrt".
        rows : np.ndarray, optional
            Row indices corresponding to vals. 
            Needs to be declared if the format for the declared Hessian is coo or csr.
        cols : np.ndarray, optional
            Column indices corresponding to vals. 
            Needs to be declared if the format for the declared Hessian is coo or csc.
        ind_ptr : np.ndarray, optional
            Index pointer array for compressed index formats.
            Needs to be declared if the format for the declared Hessian is csr or csc.
        shape : tuple, optional
            Shape in which 'vals' are going to be provided later by the user 
            (for dense or sparse formats).
        vals : float or np.ndarray, optional
            Values for the Hessian. 
            Useful if the objective gradient wrt "of" design variable is 
            independent of or linearly-dependent on the "wrt" design variables.
            'vals' are nonzero entries corresponding to 'rows' or 'cols' if the 
            declared Hessian is in sparse format.
        '''
        if (wrt not in self.design_variables_dict) or (of not in self.design_variables_dict):
            raise KeyError(f'Hessian is declared for at least one undeclared design variable in ({of}, {wrt})')

        if 'obj_hess' not in self.declared_variables:
            self.declared_variables.append('obj_hess')

        self.p2F_pxx_dict[of, wrt] = dict(
            vals=vals,
            rows=rows,
            cols=cols,
            ind_ptr=ind_ptr,
            vals_shape=shape,
        )

    def declare_lagrangian_hessian(self,
                                   of,
                                   wrt,
                                   shape=None,
                                   vals=None,
                                   rows=None,
                                   cols=None,
                                   ind_ptr=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare nonzero Lagrangian Hessian components L_{xy} = d^2L/dydx.
        Hessian components that are undeclared are assumed to be zeros.
        If the Hessian component is provided later (in compute_lagrangian_hessian() method) 
        in one of the sparse formats coo, csr, or csc,
        declare the Hessian sparsity by calling this method with kwargs (rows, cols), (rows, ind_ptr), 
        or (cols, ind_ptr), respectively.

        Parameters
        ----------
        of : str
            Name of the variable x in F_{xy}.
            Note that the declared Hessian is the derivative of
            the Lagrangian gradient "dL/dx" with respect to y, hence the keyword "of".
        wrt : str
            Name of the variable y in F_{xy}.
            Note that the declared Hessian is the derivative of
            the Lagrangian gradient dL/dx with respect to "y", hence the keyword "wrt".
        rows : np.ndarray, optional
            Row indices corresponding to vals. 
            Needs to be declared if the format for the declared Hessian is coo or csr.
        cols : np.ndarray, optional
            Column indices corresponding to vals. 
            Needs to be declared if the format for the declared Hessian is coo or csc.
        ind_ptr : np.ndarray, optional
            Index pointer array for compressed index formats.
            Needs to be declared if the format for the declared Hessian is csr or csc.
        shape : tuple, optional
            Shape in which 'vals' are going to be provided later by the user 
            (for dense or sparse formats).
        vals : float or np.ndarray, optional
            Values for the Lagrangian Hessian. 
            Useful if the Lagrangian gradient wrt "of" design variable is 
            independent of or linearly-dependent on the "wrt" design variables.
            'vals' are nonzero entries corresponding to 'rows' or 'cols' if the 
            declared Hessian is in sparse format.
        '''
        if (wrt not in self.design_variables_dict) or (of not in self.design_variables_dict):
            raise KeyError(f'Lagrangian Hessian is declared for at least one undeclared design variable in ({of}, {wrt})')

        if 'lag_hess' not in self.declared_variables:
            self.declared_variables.append('lag_hess')

        self.p2L_pxx_dict[of, wrt] = dict(
            vals=vals,
            rows=rows,
            cols=cols,
            ind_ptr=ind_ptr,
            vals_shape=shape,
        )

    def declare_objective_hvp(self, wrt, vals=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare objective Hessian-vector product (HVP).

        Parameters
        ----------
        wrt : str
            Name of the variables w.r.t. which the HVP needs to be declared.
        vals : float or np.ndarray, optional
            Values for the objective HVP. Useful if the HVP 
            is constant w.r.t. the declared "wrt" design variables.
        '''
        # Note 'of' and 'wrt' are same here since the Hessian is symmetric 
        # Technically, it should be 'of' to maintain parallelism with the JVP declaration
        if wrt not in self.design_variables_dict:
            raise KeyError(f'HVP is declared with respect to undeclared design variable {wrt}')
        
        if 'obj_hvp' not in self.declared_variables:
            self.declared_variables.append('obj_hvp')

        if vals is not None:
            self.obj_hvp[wrt] = vals

    def declare_lagrangian_hvp(self, wrt, vals=None,):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare Lagrangian Hessian-vector product (HVP).

        Parameters
        ----------
        wrt : str
            Name of the variables w.r.t. which the HVP needs to be declared.
        vals : float or np.ndarray, optional
            Values for the Lagrangian HVP. Useful if the HVP 
            is constant w.r.t. the declared "wrt" design variables.
        '''
        if wrt not in self.design_variables_dict:
            raise KeyError(f'HVP is declared with respect to undeclared design variable {wrt}')
        
        if 'lag_hvp' not in self.declared_variables:
            self.declared_variables.append('lag_hvp')

        if vals is not None:
            self.lag_hvp[wrt] = vals

    # USER-DEFINED COMPUTE METHODS BELOW:
    # ===================================

    def compute_objective(self, dvs, obj):
        """
        Compute the objective function given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        obj : dict
            Objective function name and value.
        """
        pass

    def compute_constraints(self, dvs, con):
        """
        Compute the constraint vector given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        con : array_manager.Vector
            Vector of constraints.
            This abstract vector has dictionary-type views for 
            component constraint vectors.
        """
        pass

    def compute_lagrangian(self, dvs, lag_mult, lag):
        """
        Compute the Lagrangian given the design variable and Lagrange multiplier vectors.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        lag_mult : array_manager.Vector
            Vector of Lagrange multipliers.
            This abstract vector has dictionary-type views for 
            component Lagrange multiplier vectors corresponding to
            component constraints.
        lag : dict
            Objective function name and value.
        """
        pass

    def compute_objective_gradient(self, dvs, grad):
        """
        Compute the objective function gradient given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        grad : array_manager.Vector
            Gradient vector of the objective function with respect to the design variable vector.
            This abstract vector has dictionary-type views for component gradient vectors 
            corresponding to component design variable vectors.
            
        """
        pass

    def compute_lagrangian_gradient(self, dvs, lag_mult, lag_grad):
        """
        Compute the Lagrangian gradient given the design variable and Lagrange multiplier vectors.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        lag_mult : array_manager.Vector
            Vector of Lagrange multipliers.
            This abstract vector has dictionary-type views for 
            component Lagrange multiplier vectors corresponding to
            component constraints.
        lag_grad : array_manager.Vector
            Gradient vector of the Lagrangian with respect to the design variable vector.
            This abstract vector has dictionary-type views for component gradient vectors 
            corresponding to component design variable vectors.
        """
        pass

    def compute_constraint_jacobian(self, dvs, jac):
        """
        Compute the constraint Jacobian with respect to the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        jac : array_manager.Matrix
            Jacobian matrix of the constraints with respect to the design variable vector.
            This abstract matrix has dictionary-type views for 
            component sub-Jacobians with keys (of,wrt) where 
            'of' is the constraint name and 'wrt' the design variable name.
        """
        pass

    def compute_constraint_jvp(self, dvs, vec, jvp):
        """
        Compute the constraint Jacobian-vector product (JVP) for the given design variable vector and
        multiplying vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        vec : array_manager.Vector
            Vector to multiply with the Jacobian.
            This abstract vector has dictionary-type views corresponding to 
            component design variable vectors.
        jvp : array_manager.Vector
            Constraint Jacobian-vector product.
            This abstract vector has dictionary-type views corresponding to 
            component constraint vectors.
        """
        pass

    def compute_constraint_vjp(self, dvs, vec, vjp):
        """
        Compute the constraint vector-Jacobian product (VJP) for the given design variable vector and
        multiplying vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        vec : array_manager.Vector
            Vector to multiply with the Jacobian.
            This abstract vector has dictionary-type views corresponding to 
            component constraint vectors.
        vjp : array_manager.Vector
            Constraint vector-Jacobian product.
            This abstract vector has dictionary-type views corresponding to 
            component design variable vectors.
        """
        pass

    def compute_objective_hessian(self, dvs, obj_hess):
        """
        Compute the objective Hessian given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        obj_hess : array_manager.Matrix
            Hessian matrix of the objective function with respect to the design variable vector.
            This abstract matrix has dictionary-type views for 
            component sub-Hessians F_xy = d2F/dydx with keys (of,wrt) where 
            'of' is the x design variable name and 
            'wrt' is the y design variable name.
        """
        pass

    def compute_lagrangian_hessian(self, dvs, lag_mult, lag_hess):
        """
        Compute the Lagrangian Hessian given the design variable and Lagrange multiplier vectors.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        lag_mult : array_manager.Vector
            Vector of Lagrange multipliers.
            This abstract vector has dictionary-type views for 
            component Lagrange multiplier vectors corresponding to
            component constraints.
        lag_hess : array_manager.Matrix
            Hessian matrix of the Lagrangian with respect to the design variable vector.
            This abstract matrix has dictionary-type views for 
            component sub-Hessians L_xy = d2L/dydx with keys (of,wrt) where 
            'of' is the x design variable name and 
            'wrt' is the y design variable name.
        """
        pass
    
    def compute_objective_hvp(self, dvs, vec, obj_hvp):
        """
        Compute the objective Hessian-vector product (HVP) for a given design variable vector and
        a multiplying vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        vec : array_manager.Vector
            Vector to multiply with the Hessian.
            This abstract vector has dictionary-type views corresponding to 
            component design variable vectors.
        obj_hvp : array_manager.Vector
            Objective Hessian-vector product.
            This abstract vector has dictionary-type views corresponding to 
            component design variable vectors.
        """
        pass

    def compute_lagrangian_hvp(self, dvs, lag_mult, vec, lag_hvp):
        """
        Compute the Lagrangian Hessian-vector product (HVP) for a given design variable vector,
        Lagrange multiplier vector, and multiplying vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        lag_mult : array_manager.Vector
            Vector of Lagrange multipliers.
            This abstract vector has dictionary-type views for 
            component Lagrange multiplier vectors corresponding to
            component constraints.
        vec : array_manager.Vector
            Vector to multiply with the Lagrangian Hessian.
            This abstract vector has dictionary-type views corresponding to 
            component design variable vectors.
        lag_hvp : array_manager.Vector
            Lagrangian Hessian-vector product.
            This abstract vector has dictionary-type views corresponding to 
            component design variable vectors.
        """
        # TODO: fill this
        pass

    # # SEPARATE CONSTRAINT HESSIANS AND HVP COMPUTATION WILL NOT BE SUPPORTED FOR MEMORY REASONS.
    # # This can be indirectly computed inside compute_lagrangian_hessian and compute_lagrangian_hvp()
    # def compute_constraint_hessian(self, dvs, idx, con_hess):
    #     """
    #     Compute the constraint Hessian given the design variable vector and the index of the constraint.

    #     Parameters
    #     ----------
    #     dvs : array_manager.Vector
    #         Design variable vector.
    #     idx : int
    #         Index of the constraint.
    #     con_hess : array_manager.Matrix
    #         Hessian matrix of the specified idx-th constraint with respect to the design variable vector.
    #     """
    #     pass

    # def compute_constraint_hvp(self, dvs, idx, v, con_hvp):
    #     """
    #     Compute the constraint Hessian-vector product for a given design variable vector,
    #     a multiplying vector, and the index of the constraint.

    #     Parameters
    #     ----------
    #     dvs : array_manager.Vector
    #         Design variable vector.
    #     idx : int
    #         Index of the constraint.
    #     v : array_manager.Vector
    #         Vector to multiply with the Hessian. 
    #     con_hvp : array_manager.Vector
    #         Constraint Hessian-vector product.
    #     """
    #     pass


    # Finite Difference Approximations of Derivatives:
    # ================================================
    def use_finite_differencing(self, derivative, step=1e-6):
        '''
        User calls this method within compute methods to approximate derivatives.

        Parameters
        ----------
        derivative : str
            Derivative to approximate.
            Valid options: 'objective_gradient', 'objective_hessian', 'constraint_jacobian',
            'objective_hvp', 'constraint_jvp'.
        step : float or np.ndarray, default=1e-6
            Finite difference step size.
        '''
        if np.isscalar(step):
            if not np.isreal(step):
                raise ValueError('Step size "step" must be a real number.')
        elif step.shape != (self.nx,):
            raise ValueError('Step size "step" must be a scalar or an array of size (nx,) where nx is the number of design variables.')
        if derivative == 'objective_gradient':
            x = self.x.get_data()
            self.compute_objective(self.x, self.obj)
            f0 = list(self.obj.values())[0]
            g_fd = np.zeros((self.nx,))
            for i in range(self.nx):
                e = np.zeros((self.nx,))
                e[i] = 1.
                self.x.set_data(x + step*e)
                self.compute_objective(self.x, self.obj)
                f1 = list(self.obj.values())[0]
                g_fd[i] = (f1 - f0)
            g_fd /= step
            self.pF_px.set_data(g_fd)
            
        elif derivative == 'objective_hessian':
            x = self.x.get_data()
            self.compute_objective_gradient(self.x, self.pF_px)
            g0 = self.pF_px.get_data()
            H_fd = np.zeros((self.nx, self.nx))
            for i in range(self.nx):
                e = np.zeros((self.nx,))
                e[i] = 1.
                self.x.set_data(x + step*e)
                self.compute_objective_gradient(self.x, self.pF_px)
                g1 = self.pF_px.get_data()
                H_fd[:, i] = (g1 - g0)
            H_fd /= step # rowwise division if step is a 1d array
            self.p2F_pxx.vals.set_data(H_fd.flatten())

        elif derivative == 'constraint_jacobian':
            x = self.x.get_data()
            self.compute_constraints(self.x, self.con)
            c0 = self.con.get_data()
            J_fd = np.zeros((self.nc, self.nx))
            for i in range(self.nx):
                e = np.zeros((self.nx,))
                e[i] = 1.
                self.x.set_data(x + step*e)
                self.compute_constraints(self.x, self.con)
                c1 = self.con.get_data()
                J_fd[:, i] = (c1 - c0)
            J_fd /= step # rowwise division if step is a 1d array
            self.pC_px.vals.set_data(J_fd.flatten())
        
        elif not np.isscalar(step):
            raise ValueError('Step size "step" must be a scalar for JVP or HVP derivative.')

        elif derivative == 'objective_hvp':
            x = self.x.get_data()
            self.compute_objective_gradient(self.x, self.pF_px)
            g0 = self.pF_px.get_data()
            v = self.vec_hvp.get_data()

            self.x.set_data(x + step*v)
            self.compute_objective_gradient(self.x, self.pF_px)
            g1 = self.pF_px.get_data()
            hvp_fd = (g1 - g0) / step
            self.obj_hvp.set_data(hvp_fd)

        elif derivative == 'constraint_jvp':
            x = self.x.get_data()
            self.compute_constraints(self.x, self.con)
            c0 = self.con.get_data()
            v = self.vec_jvp.get_data()

            self.x.set_data(x + step*v)
            self.compute_constraints(self.x, self.con)
            c1 = self.con.get_data()
            jvp_fd = (c1 - c0) / step
            self.jvp.set_data(jvp_fd)

        else:
            raise NotImplementedError('Finite differencing is not implemented for the requested derivative.')


    # WRAPPER FOR USER-DEFINED COMPUTE METHODS BELOW (USED BY Optimizer() OBJECTS):
    # =============================================================================
    
    def _compute_objective(self, x):
        '''
        Wrapper for user-defined compute_objective(). 
        Argument here is a numpy array as opposed to array_manager.Vector. 
        Performs problem- and optimizer-independent scaling before passing
        objective to optimizers in modOpt.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.

        Returns
        -------
        float
            Objective function value.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_objective(self.x, self.obj)
        objectives = np.array(list(self.obj.values()))
        objective_scalers = np.array(list(self.obj_scaler.values()))
        if isinstance(objectives[0], np.ndarray):
            return (objectives[0] * objective_scalers[0])[0]
        return (objectives[0] * objective_scalers[0])
    
    def _compute_lagrangian(self, x, z, auto=False):
        '''
        Wrapper for user-defined compute_lagrangian(). 
        Arguments here are numpy arrays as opposed to array_manager.Vector. 
        Performs problem- and optimizer-independent scaling before passing
        Lagrangian to optimizers in modOpt.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        z : np.ndarray
            Lagrange multiplier vector.
        auto : bool, default=False
            auto=True implies the Problem() object automatically computes the
            Lagrangian for the optimizer if the user has not
            implemented a compute_lagrangian() method.

        Returns
        -------
        float
            Lagrangian function value.
        '''
        if auto:
            return self._compute_objective(x) - np.inner(z.flatten(), self._compute_constraints(x).flatten())
        self.x.set_data(x / self.x_scaler)
        objective_scaler = list(self.obj_scaler.values())[0]
        self.lag_mult.set_data(z*self.c_scaler/objective_scaler)
        self.compute_lagrangian(self.x, self.lag_mult, self.lag)
        return self.lag * objective_scaler

    def _compute_objective_gradient(self, x):
        '''
        Wrapper for user-defined compute_objective_gradient(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.

        Returns
        -------
        np.ndarray
            1-dimensional objective gradient vector.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_objective_gradient(self.x, self.pF_px)
        # print('grad', self.pF_px.get_data())
        objective_scaler = list(self.obj_scaler.values())[0]
        return self.pF_px.get_data() * objective_scaler / self.x_scaler
    
    def _compute_lagrangian_gradient(self, x, z, auto=False):
        '''
        Wrapper for user-defined compute_lagrangian_gradient(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        z : np.ndarray
            Lagrange multiplier vector.
        auto : bool, default=False
            auto=True implies Problem() object automatically computes the
            Lagrangian for the optimizer if the user has not
            implemented a compute_lagrangian() method.

        
        Returns
        -------
        np.ndarray
            1-dimensional Lagrangian gradient vector.
        '''
        if auto:
            g = self._compute_objective_gradient(x)
            J  = self._compute_constraint_jacobian(x)
            return  g - J.T @ z.flatten()
        self.x.set_data(x / self.x_scaler)
        objective_scaler = list(self.obj_scaler.values())[0]
        self.lag_mult.set_data(z*self.c_scaler/objective_scaler)
        self.compute_lagrangian_gradient(self.x, self.lag_mult, self.pL_px)
        return self.pL_px.get_data() * objective_scaler / self.x_scaler

    def _compute_objective_hessian(self, x):
        '''
        Wrapper for user-defined compute_objective_hessian(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
                
        Returns
        -------
        np.ndarray
            2-dimensional objective Hessian matrix 
            (sparse or dense depending on self.options['hess_format']).
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_objective_hessian(self.x, self.p2F_pxx)
        self.obj_hess.update_bottom_up()
        objective_scaler = list(self.obj_scaler.values())[0]
        # return self.obj_hess.get_std_array() * objective_scaler / np.outer(self.x_scaler, self.x_scaler)
        x_scaler_row = self.x_scaler.reshape(1, self.x_scaler.size)
        return self.obj_hess.get_std_array() * (objective_scaler / x_scaler_row) / x_scaler_row.T
    
    def _compute_lagrangian_hessian(self, x, z):
        '''
        Wrapper for user-defined compute_lagrangian_hessian(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        z : np.ndarray
            Lagrange multiplier vector.
                        
        Returns
        -------
        np.ndarray
            2-dimensional Lagrangian Hessian matrix 
            (sparse or dense depending on self.options['hess_format']).
        '''
        self.x.set_data(x / self.x_scaler)
        objective_scaler = list(self.obj_scaler.values())[0]
        self.lag_mult.set_data(z*self.c_scaler/objective_scaler)
        self.compute_lagrangian_hessian(self.x, self.lag_mult, self.p2L_pxx)
        self.lag_hess.update_bottom_up()
        # return self.lag_hess.get_std_array() * objective_scaler / np.outer(self.x_scaler, self.x_scaler)
        x_scaler_row = self.x_scaler.reshape(1, self.x_scaler.size)
        return self.lag_hess.get_std_array() * (objective_scaler / x_scaler_row) / x_scaler_row.T

    def _compute_objective_hvp(self, x, v):
        '''
        Wrapper for user-defined compute_objective_hvp(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        v : np.ndarray
            Vector to right-multiply Hessian with.
                        
        Returns
        -------
        np.ndarray
            1-dimensional objective HVP vector.
        '''
        self.x.set_data(x / self.x_scaler)
        self.vec_hvp.set_data(v/self.x_scaler)
        self.compute_objective_hvp(self.x, self.vec_hvp, self.obj_hvp)
        objective_scaler = list(self.obj_scaler.values())[0]
        return self.obj_hvp.get_data() * objective_scaler / self.x_scaler
    
    def _compute_lagrangian_hvp(self, x, z, v):
        '''
        Wrapper for user-defined compute_lagrangian_hvp(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        z : np.ndarray
            Lagrange multiplier vector.
        v : np.ndarray
            Vector to right-multiply Hessian with.
                        
        Returns
        -------
        np.ndarray
            1-dimensional Lagrangian HVP vector.
        '''
        self.x.set_data(x / self.x_scaler)
        objective_scaler = list(self.obj_scaler.values())[0]
        self.lag_mult.set_data(z*self.c_scaler/objective_scaler)
        self.vec_hvp.set_data(v/self.x_scaler)
        self.compute_lagrangian_hvp(self.x, self.lag_mult, self.vec_hvp, self.lag_hvp)
        return self.lag_hvp.get_data() * objective_scaler / self.x_scaler

    def _compute_constraints(self, x):
        '''
        Wrapper for user-defined compute_constraints(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
                        
        Returns
        -------
        np.ndarray
            1-dimensional constraint vector.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_constraints(self.x, self.con)
        # print('con', self.con.get_data())
        return self.con.get_data() * self.c_scaler

    def _compute_constraint_jacobian(self, x):
        '''
        Wrapper for user-defined compute_constraint_jacobian(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
                        
        Returns
        -------
        np.ndarray
            2-dimensional constraint Jacobian matrix.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_constraint_jacobian(self.x, self.pC_px)
        self.jac.update_bottom_up()
        # print('jac', self.jac.get_std_array())
        return self.jac.get_std_array() * np.outer(self.c_scaler, 1./self.x_scaler)
    
    def _compute_constraint_jvp(self, x, v):
        '''
        Wrapper for user-defined compute_constraint_jvp(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        v : np.ndarray
            Vector to right-multiply Jacobian with.
                        
        Returns
        -------
        np.ndarray
            1-dimensional constraint JVP vector.
        '''
        self.x.set_data(x / self.x_scaler)
        self.vec_jvp.set_data(v/self.x_scaler)
        self.compute_constraint_jvp(self.x, self.vec_jvp, self.jvp)
        # print('jvp', self.jvp.get_data())
        return self.jvp.get_data() * self.c_scaler
    
    def _compute_constraint_vjp(self, x, v):
        '''
        Wrapper for user-defined compute_constraint_vjp(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        v : np.ndarray
            Vector to left-multiply Jacobian with.
                        
        Returns
        -------
        np.ndarray
            1-dimensional constraint VJP vector.
        '''
        self.x.set_data(x / self.x_scaler)
        self.vec_vjp.set_data(v*self.c_scaler)
        self.compute_constraint_vjp(self.x, self.vec_vjp, self.vjp)
        # print('vjp', self.vjp.get_data())
        return self.vjp.get_data() / self.x_scaler
    
    # # SEPARATE CONSTRAINT HESSIANS AND HVP COMPUTATION WILL NOT BE SUPPORTED FOR MEMORY REASONS.
    # # This can be indirectly computed inside compute_lagrangian_hessian and compute_lagrangian_hvp()
    # def _compute_constraint_hessian(self, x, idx):
    #     self.x.set_data(x / self.x_scaler)
    #     self.compute_constraint_hessian(self.x, idx, self.p2C_pxx)  # TODO: define self.p2C_pxx
    #     self.con_hess.update_bottom_up()                            # TODO: define self.con_hess
    #     c_scaler = self.c_scaler[idx]
    #     return self.con_hess.get_std_array() * c_scaler / np.outer(self.x_scaler, self.x_scaler)

    # def _compute_constraint_hvp(self, x, idx, v):
    #     self.x.set_data(x / self.x_scaler)
    #     self.compute_constraint_hvp(self.x, idx, v/self.x_scaler, self.con_hvp) # TODO: define self.con_hvp
    #     c_scaler = self.c_scaler[idx]
    #     return self.con_hvp.get_data() * c_scaler / self.x_scaler


    ###############################################################################
    # Everything below is applicable only for the SURF algorithm for optimization #
    ###############################################################################

    def add_state_variables(self,
                            name,
                            shape=(1, ),
                            lower=None,
                            upper=None,
                            equals=None,
                            vals=None):
        if vals is None:
            vals == np.zeros(shape)
        self.state_variables_dict[name] = dict(shape=shape,
                                               lower=lower,
                                               upper=upper,
                                               equals=equals,
                                               vals=vals)

        self.ny += np.prod(shape)

    def add_residuals(self, name, shape=(1, )):
        self.residuals_dict[name] = dict(shape=shape,
                                         lower=None,
                                         upper=None,
                                         equals=np.zeros(shape))

        self.nr += np.prod(shape)

    def declare_pF_px_gradient(self, wrt, shape=(1, ), vals=None):

        if wrt not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which gradient of F is declared'
                .format(wrt))

        else:
            self.pF_px[wrt] = vals

    def declare_pF_py_gradient(self, wrt, shape=(1, ), vals=None):

        if wrt not in self.state_variables_dict:
            raise Exception(
                'Undeclared state variable {} with respect to which gradient of F is declared'
                .format(wrt))

        else:
            self.pF_py[wrt] = vals

    def declare_pR_px_jacobian(self,
                               of,
                               wrt,
                               shape=(1, 1),
                               vals=None,
                               rows=None,
                               cols=None,
                               ind_ptr=None):

        if of not in self.residuals_dict:
            raise Exception(
                'Undeclared residual {} for which partial is declared'.
                format(of))
        elif wrt not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pR_px_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def declare_pR_py_jacobian(self,
                               of,
                               wrt,
                               shape=(1, 1),
                               vals=None,
                               rows=None,
                               cols=None,
                               ind_ptr=None):

        if of not in self.residuals_dict:
            raise Exception(
                'Undeclared residual {} for which partial is declared'.
                format(of))
        elif wrt not in self.state_variables_dict:
            raise Exception(
                'Undeclared state variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pR_py_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def declare_pC_px_jacobian(self,
                               of,
                               wrt,
                               shape=(1, 1),
                               vals=None,
                               rows=None,
                               cols=None,
                               ind_ptr=None):

        if of not in self.constraints_dict:
            raise Exception(
                'Undeclared constraint {} for which partial is declared'
                .format(of))
        elif wrt not in self.design_variables_dict:
            raise Exception(
                'Undeclared design variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pC_px_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    def declare_pC_py_jacobian(self,
                               of,
                               wrt,
                               shape=(1, 1),
                               vals=None,
                               rows=None,
                               cols=None,
                               ind_ptr=None):

        if of not in self.constraints_dict:
            raise Exception(
                'Undeclared constraint {} for which partial is declared'
                .format(of))
        elif wrt not in self.state_variables_dict:
            raise Exception(
                'Undeclared state variable {} with respect to which partial is declared'
                .format(wrt))

        else:
            self.pC_py_dict[of, wrt] = dict(
                vals=vals,
                rows=rows,
                cols=cols,
                ind_ptr=ind_ptr,
                vals_shape=shape,
            )

    # # Override for specific problems if bounds are not available in this format, eg. csdl
    # def declare_variable_bounds(self, x_lower, x_upper):
    #     self.x_lower = x_lower
    #     self.x_upper = x_upper

    # # Override for specific problems if bounds are not available in this format, eg. csdl
    # def declare_constraint_bounds(self, c_lower, c_upper):
    #     self.c_lower = c_lower
    #     self.c_upper = c_upper

    def compute_functions(self, x):
        self.x = x
        self.y = self.solve_residual_equations(
            x)  # Note: assumes a single set of residual equations
        self.f_0 = self.evaluate_objective(x, self.y)
        self.c_0 = self.evaluate_constraints(x, self.y)

        return self.f_0, self.c_0

    def compute_direct_vector(self, x):
        if self.x.get_data() != x:
            self.x.set_data(x)
            self.y = self.solve_residual_equations(
                x)  # Note: assumes a single set of residual equations

        else:
            self.y = self.solve_residual_equations(
                x, self.y
            )  # uses the previous approximation of y to warm start the nonlinear solver

        pR_px, self.pR_py = self.evaluate_residual_jacobian(
            x,
            self.y)  # Note: assumes a single set of residual equations
        self.dy_dx_0 = np.linalg.solve(self.pR_py, pR_px)

        return self.dy_dx_0

    def compute_adjoint_vector(self, x, pFun_py):
        if self.x.get_data() != x:
            self.x.set_data(x)
            self.y = self.solve_residual_equations(
                x)  # Note: assumes a single set of residual equations

        else:
            self.y = self.solve_residual_equations(
                x, self.y
            )  # uses the previous approximation of y to warm start the nonlinear solver

        pR_px, self.pR_py = self.evaluate_residual_jacobian(
            x,
            self.y)  # Note: assumes a single set of residual equations
        self.df_dr_0 = np.linalg.solve(-self.pR_py.T, pFun_py).T

        return self.df_dr_0, pR_px

    # def compute_objective_gradient(self, x):
    #     if self.x.get_data() != x:
    #         self.x.set_data(x)
    #         self.y = self.solve_residual_equations(
    #             x)  # Note: assumes a single set of residual equations

    #     else:
    #         self.y = self.solve_residual_equations(
    #             x, self.y
    #         )  # uses the previous approximation of y to warm start the nonlinear solver
    #         # Note: assumes a single set of residual equations
    #     pF_px_0, pF_py_0 = self.evaluate_objective_gradient(
    #         self, x, self.y)

    #     df_dr_0, pR_px = self.compute_adjoint_vector(x, pF_py_0)
    #     self.pF_px_0 = pF_px_0 + np.matmul(df_dr_0, pR_px)

    #     return self.pF_px_0

    # def compute_constraint_jacobian(self, x):
    #     if self.x.get_data() != x:
    #         self.x.set_data(x)
    #         self.y = self.solve_residual_equations(
    #             x)  # Note: assumes a single set of residual equations

    #     else:
    #         self.y = self.solve_residual_equations(
    #             x, self.y
    #         )  # uses the previous approximation of y to warm start the nonlinear solver
    #         # Note: assumes a single set of residual equations
    #     pC_px_0, pC_py_0 = self.evaluate_constraint_jacobian(
    #         self, x, self.y)

    #     if self.nc <= self.nr:
    #         dc_dr_0, pR_px = self.compute_adjoint_vector(x, pC_py_0)
    #         self.pC_px_0 = pC_px_0 + np.matmul(dc_dr_0, pR_px)

    #     else:
    #         dy_dx = self.compute_direct_vector(x)
    #         self.pC_px_0 = pC_px_0 - np.matmul(pC_py_0, dy_dx)

    #     return self.pC_px_0

    def evaluate_residuals(self, x, y):
        """
        Evaluate the residual vector given the design and state vectors.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        res : np.ndarray
            Vector of residuals.
        """
        pass


    def solve_residual_equations(self, x, y=None, tol=None):
        """
        Solve the implicit functions that define the residuals to compute an inexact state vector with given tolerances.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        tol : np.ndarray
            Tolerance vector.

        Returns
        -------
        inexact_y : np.ndarray
            Inexact state variable vector.
        """
        pass

    def evaluate_residual_jacobian(self, x, y):
        """
        Evaluate the residual Jacobian with respect to the state and design variable vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.

        Returns
        -------
        pR_px : np.ndarray
            Jacobian matrix of the residual functions with respect to the design vector.
        pR_py : np.ndarray
            Jacobian matrix of the residual functions with respect to the state vector.
        """
        pass

    def evaluate_lagrangian_hessian(self, x, y, lag_mult):
        """
        Evaluate the Lagrangian Hessian given the design and state vectors along with the Lagrange multiplier vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        lag_mult : np.ndarray
            Lagrange multiplier vector.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the Lagrangian function with respect to the design and state vectors.
        """
        hessian = self.evaluate_objective_hessian(x, y)

        # We cannot afford to store all the constraint Hessians for vectorizing the computation of Lagrangian Hessian so 'for loop' is unavoidable.
        for i in range(len(lag_mult)):
            hessian += lag_mult[i] * self.evaluate_constraint_hessian(
                x, y, i)

        return hessian

    def evaluate_adjoint_vector(self, ):
        pass

    def evaluate_penalty_hessian(self, x, y, rho):
        """
        Evaluate the penalty Hessian given the design and state vectors along with the Lagrange multiplier vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        rho : np.ndarray
            Vector of penalty parameters.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the penalty function with respect to the design and state vectors.
        """
        hessian = self.evaluate_objective_hessian(x, y)

        # We cannot afford to store all the constraint Hessians for vectorizing the computation of Lagrangian Hessian so a 'for loop' is unavoidable.
        for i in range(len(rho)):
            hessian += rho[i] * self.evaluate_constraint_hessian(
                x, y, i)

        return hessian

    def evaluate_augmented_lagrangian_hessian(self, x, y, rho,
                                              lag_mult):
        """
        Evaluate the augmented Lagrangian Hessian given the design and state vectors along with the Lagrange multiplier vector.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        rho: np.ndarray
            Vector of penalty parameters
        lag_mult : np.ndarray
            Lagrange multiplier vector.

        Returns
        -------
        hessian : np.ndarray
            Hessian matrix of the augmented Lagrangian function with respect to the design and state vectors.
        """
        hessian = self.evaluate_objective_hessian(x, y)

        # We cannot afford to store all the constraint Hessians for vectorizing the computation of Lagrangian Hessian so 'for loop' is unavoidable.
        for i in range(len(lag_mult)):
            hessian += lag_mult[i] * self.evaluate_constraint_hessian(
                x, y, i)

        return hessian

    # With Hessian-vector products, we can also compute products with the augmented Lagrangian Hessian
    def evaluate_hvp(self, x, y, lag_mult, rho, vx, vy):
        """
        Evaluate the Hessian-vector product along the direction vector specified, for given design and state vectors, for a given 
        Hessian (objective, penalty, Lagrangian, or Augmneted Lagrangian) specified by the Lagrange multipliers and/or penalty parameters.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        vx : np.ndarray
            x block of vector to be right-multiplied.
        vy : np.ndarray
            y block of vector to be right-multiplied.
        rho : np.ndarray
            Vector of penalty parameters.
        lag_mult : np.ndarray
            Lagrange multiplier vector.

        Returns
        -------
        wx : np.ndarray
            x block of the Hessian-vector product.
        wy : np.ndarray
            y block of the Hessian-vector product.
        """
        pass