import numpy as np
import scipy as sp
import warnings
from modopt import Problem as Problem
from modopt.core.recording_and_hotstart import hot_start, record

try:
    import pycutest
except:
    warnings.warn("'pycutest' could not be imported")

# Note: modopt.CUTEstProblem is different from pycutest.CUTEstProblem
class CUTEstProblem(Problem):
    '''
    Class that wraps pyCUTEst problems for modOpt.

    General functionality:
    find_problems(): Find problems that satisfy given criteria as per CUTEst classification scheme.
    problem_properties(): Get properties of a given problem.
    print_available_sif_params(): Print available optional input parameters for a given problem.
    import_problem(): Import a given problem with optional parameters. Returns a pycutest.CUTEstProblem object.

    '''
    def initialize(self, ):
        '''
        Initialize the Problem() instance for a CUTEstProblem.
        '''
        self.options.declare('cutest_problem', types=pycutest.CUTEstProblem)

    def setup(self, ):
        '''
        Setup the problem name, initial guess, and problem dimensions.
        '''
        prob = self.options['cutest_problem']
        # Set problem_name
        # (Since it is not constructed explicitly using the Problem() class)
        self.problem_name = prob.name
        self.problem_properties = pycutest.problem_properties(self.problem_name)
        self.x0 = prob.x0 * 1.

        # Set problem dimensions
        self.nx = prob.n
        self.nc = prob.m
        if self.nc > 0:
            self.constrained = True

        self.model_evals = 0                      # number of function evaluations
        self.deriv_evals = 0                      # number of derivative evaluations
        # self.fail1 = False                      # failure of functions
        # self.fail2 = False                      # failure of functions or derivatives
        self.warm_x         = self.x0 - 1.      # (x0 - 1.) to keep it different from initial dv values
        self.warm_x_deriv   = self.x0 - 2.      # (x0 - 2.) to keep it different from initial dv and warm_x values

        # Other attributes available in pycutest.CUTEstProblem
        self.sifParams = prob.sifParams         # dict of parameters passed to sifdecode
        self.sifOptions = prob.sifOptions       # list of extra options passed to sifdecode
        self.vartype = prob.vartype             # array of variable types (NumPy array size n, entry vartype[i] 
                                                # indicates that x[i] is real(0), boolean(1), or integer(2))
        self.nnzh = prob.nnzh                   # number of nonzero entries in upper triangular part of objective Hessian 
                                                # (for all variables, including fixed)
        self.nonlinear_vars_first = prob.nonlinear_vars_first # flag if all nonlinear variables are listed before linear variables
        self.n_full = prob.n_full               # total number of variables in CUTEst problem (n_free + n_fixed)
        self.n_free = prob.n_free               # number of free variables in CUTEst problem
        self.n_fixed = prob.n_fixed             # number of fixed variables in CUTEst problem

        if self.constrained:
            self.eq_cons_first = prob.eq_cons_first         # flag if all equality constraints are listed before inequality constraints
            self.linear_cons_first = prob.linear_cons_first # flag if all linear constraints are listed before nonlinear constraints
            self.nnzj = prob.nnzj                           # number of nonzero entries in constraint Jacobian (for all variables, including fixed)
            self.v0   = prob.v0                             # initial values of the Lagrange multipliers (np.array of shape (m,))
            self.is_eq_cons = prob.is_eq_cons               # shape (m,) np.array of Boolean flags indicating if i-th constraint is 
                                                            # equality or not (i.e. inequality)
            self.is_linear_cons = prob.is_linear_cons       # shape (m,) np.array of Boolean flags indicating if i-th constraint is 
                                                            # linear or not (i.e. nonlinear)

        # Create list of user-defined callbacks
        self.user_defined_callbacks = []        # list of user-defined callbacks
        if self.problem_properties['objective'] != 'none':
            self.user_defined_callbacks += ['obj']
            if self.problem_properties['degree'] > 0:
                self.user_defined_callbacks += ['grad']
                if self.problem_properties['degree'] > 1:
                    self.user_defined_callbacks += ['obj_hess']
                    if not self.constrained:
                        self.user_defined_callbacks += ['obj_hvp']
        if self.constrained:
            self.user_defined_callbacks += ['con']
            self.user_defined_callbacks += ['lag']
            if self.problem_properties['degree'] > 0:
                self.user_defined_callbacks += ['jac']
                self.user_defined_callbacks += ['lag_grad']
                self.user_defined_callbacks += ['jvp']
                self.user_defined_callbacks += ['vjp']
                if self.problem_properties['degree'] > 1:
                    self.user_defined_callbacks += ['lag_hess']
                    self.user_defined_callbacks += ['lag_hvp']

        self.declared_variables = ['dv'] + self.user_defined_callbacks

    def _setup_bounds(self):
        '''
        Setup the variable and constraint bounds for the modOpt problem.
        '''
        prob = self.options['cutest_problem']

        # Set design variable bounds
        x_l = prob.bl * 1.
        x_u = prob.bu * 1.

        self.x_lower = np.where(x_l == -1.0e20, -np.inf, x_l)
        self.x_upper = np.where(x_u ==  1.0e20,  np.inf, x_u)

        # Set constraint bounds
        if self.nc > 0:
            self.c_lower = np.where(prob.cl == -1.0e20, -np.inf, prob.cl)
            self.c_upper = np.where(prob.cu ==  1.0e20,  np.inf, prob.cu)
        else:
            self.c_lower = None
            self.c_upper = None

    def compute_objective(self, dvs, obj):
        pass
    def compute_objective_gradient(self, dvs, grad):
        pass
    def compute_constraints(self, dvs, con):
        pass
    def compute_constraint_jacobian(self, dvs, jac):
        pass
    def compute_lagrangian(self, dvs, lmult, lag):
        pass
    def compute_lagrangian_gradient(self, dvs, lmult, lgrad):
        pass
    def compute_objective_hessian(self, dvs, hess):
        pass
    def compute_lagrangian_hessian(self, dvs, lmult, lhess):
        pass
    def compute_constraint_jvp(self, dvs, v, jvp):
        pass
    def compute_constraint_vjp(self, dvs, v, vjp):
        pass
    def compute_objective_hvp(self, dvs, v, hvp):
        pass
    def compute_lagrangian_hvp(self, dvs, lmult, v, lhvp):
        pass
    
    # TODO: Add decorators for checking if x is warm and for updating dvs
    # TODO: Add jvp, hvp, laggrad, laghess, etc. for CUTEst problems
    @record(['x'], ['obj'])
    @hot_start(['x'], ['obj'])
    def _compute_objective(self, x, force_rerun=False, check_failure=False):
        '''
        Compute the objective 'f' for the given design variable vector 'x'.
        '''
        prob = self.options['cutest_problem']
        if not np.array_equal(self.warm_x, x):
            # for unconstrained problems, objcons() returns (f, None)
            self.f, self.c = prob.objcons(x)
            self.warm_x = x * 1.
            self.model_evals += 1
        return self.f
        # return failure_flag, sim.objective()

    @record(['x'], ['grad'])
    @hot_start(['x'], ['grad'])
    def _compute_objective_gradient(self, x, force_rerun=False, check_failure=False):
        '''
        Compute the objective gradient 'g' for the given design variable vector 'x'.
        '''
        prob = self.options['cutest_problem']
        if not np.array_equal(self.warm_x_deriv, x):
            self.g, self.j = prob.lagjac(x)
            self.warm_x_deriv = x * 1.
            self.deriv_evals += 1
        return self.g

    @record(['x'], ['con'])
    @hot_start(['x'], ['con'])
    def _compute_constraints(self, x, force_rerun=False, check_failure=False):
        '''
        Compute the constraints 'c' for the given design variable vector 'x'.
        '''
        prob = self.options['cutest_problem']
        if not np.array_equal(self.warm_x, x):
            self.f, self.c = prob.objcons(x)
            self.warm_x = x * 1.
            self.model_evals += 1
        return self.c

    @record(['x'], ['jac'])
    @hot_start(['x'], ['jac'])
    def _compute_constraint_jacobian(self, x, force_rerun=False, check_failure=False):
        '''
        Compute the constraint Jacobian 'j' for the given design variable vector 'x'.
        '''
        prob = self.options['cutest_problem']
        if not np.array_equal(self.warm_x_deriv, x):
            # for unconstrained problems, lagjac() returns (g, None)
            self.g, self.j = prob.lagjac(x)
            self.warm_x_deriv = x * 1.
            self.deriv_evals += 1
        return self.j

    @record(['x'], ['failure', 'obj', 'con', 'grad', 'jac'])
    @hot_start(['x'], ['failure', 'obj', 'con', 'grad', 'jac'])
    def _compute_all(self, x, force_rerun=False, check_failure=False):
        '''
        Compute the objective, constraints, objective gradient,
        and constraint Jacobian for the given design variable vector.

        Parameters
        ----------
        x: np.ndarray
            Vector of design variable values.
        
        Returns
        -------
        self.f : float
            Objective value.
        self.c : np.ndarray
            Constraint vector.
        self.g : np.ndarray
            Objective gradient vector.
        self.j : np.ndarray
            Constraint Jacobian vector.
        '''
        prob = self.options['cutest_problem']
        if not np.array_equal(self.warm_x, x): # here, warm_x = warm_x_deriv
            self.f, self.c = prob.objcons(x)
            self.g, self.j = prob.lagjac(x)
            self.warm_x = x * 1.
            self.warm_x_deriv = x * 1.
            self.model_evals += 1
            self.deriv_evals += 1
        return False, self.f, self.c, self.g, self.j
    
    @record(['x'], ['obj_hess'])
    @hot_start(['x'], ['obj_hess'])
    def _compute_objective_hessian(self, x):
        '''
        Compute the objective Hessian 'h' for the given design variable vector 'x'.
        '''
        prob = self.options['cutest_problem']
        self.h = prob.ihess(x)
        return self.h
    
    @record(['x', 'mu'], ['lag_hess'])
    @hot_start(['x', 'mu'], ['lag_hess'])
    def _compute_lagrangian_hessian(self, x, mu):
        '''
        Compute the Lagrangian Hessian 'lh' for the given design variable vector 'x' and Lagrange multipliers 'mu'.
        '''
        prob = self.options['cutest_problem']
        self.lh = prob.hess(x, mu)
        return self.lh
    
    @record(['x', 'mu'], ['lag'])
    @hot_start(['x', 'mu'], ['lag'])
    def _compute_lagrangian(self, x, mu):
        '''
        Compute the Lagrangian 'l' for the given design variable vector 'x' and Lagrange multipliers 'mu'.
        '''
        prob = self.options['cutest_problem']
        self.l = prob.lag(x, mu)
        return self.l
    
    @record(['x', 'mu'], ['lag_grad'])
    @hot_start(['x', 'mu'], ['lag_grad'])
    def _compute_lagrangian_gradient(self, x, mu):
        '''
        Compute the Lagrangian gradient 'lg' for the given design variable vector 'x' and Lagrange multipliers 'mu'.
        '''
        prob = self.options['cutest_problem']
        g, self.lg = prob.lagjac(x, mu)
        return self.lg

    @record(['x', 'v'], ['jvp'])
    @hot_start(['x', 'v'], ['jvp'])
    def _compute_constraint_jvp(self, x, v):
        '''
        Compute the Jacobian-vector product 'jvp' for the given design variable vector 'x' and vector 'v'.
        '''
        prob = self.options['cutest_problem']
        self.jvp = prob.jprod(v, x=x)
        return self.jvp
    
    @record(['x', 'v'], ['vjp'])
    @hot_start(['x', 'v'], ['vjp'])
    def _compute_constraint_vjp(self, x, v):
        '''
        Compute the vector-Jacobian product 'vjp' for the given design variable vector 'x' and vector 'v'.
        '''
        prob = self.options['cutest_problem']
        self.vjp = prob.jprod(v, x=x, transpose=True)
        return self.vjp

    @record(['x', 'v'], ['obj_hvp'])
    @hot_start(['x', 'v'], ['obj_hvp'])
    def _compute_objective_hvp(self, x, v):
        '''
        Compute the objective Hessian-vector product 'hvp' for the given design variable vector 'x' and vector 'v'.
        Only works for UNCONSTRAINED problems. (v must be None for unconstrained problems)
        '''
        if self.constrained:
            raise ValueError("Objective Hessian-vector product is not defined for constrained CUTEST problems."\
                             "Use 'compute_lagrangian_hvp' for constrained problems.")
        prob = self.options['cutest_problem']
        self.hvp = prob.hprod(v, x=x, v=None)
        return self.hvp
    
    @record(['x', 'mu', 'v'], ['lag_hvp'])
    @hot_start(['x', 'mu', 'v'], ['lag_hvp'])
    def _compute_lagrangian_hvp(self, x, mu, v):
        '''
        Compute the Lagrangian Hessian-vector product 'lhvp' for the given design variable vector 'x', Lagrange multipliers 'mu', and vector 'v'.
        Only works for CONSTRAINED problems. (v=mu must be specified for constrained problems)
        '''
        if not self.constrained:
            raise ValueError("Lagrangian Hessian-vector product is not defined for unconstrained CUTEST problems."\
                             "Use 'compute_objective_hvp' for unconstrained problems.")
        prob = self.options['cutest_problem']
        self.lhvp = prob.hprod(v, x=x, v=mu)
        return self.lhvp

    def get_usage_statistics(self):
        '''
        Get the usage statistics for the problem.
        '''
        prob = self.options['cutest_problem']
        stats = prob.report()
        self.noev = stats['f']      # number of objective evaluations
        self.ncev = stats['c']      # number of constraint evaluations
        self.nogev = stats['g']     # number of objective gradient evaluations
        self.ncgev = stats['cg']    # number of constraint gradient evaluations
        self.nohev = stats['H']     # number of objective Hessian evaluations
        self.nchev = stats['cH']    # number of constraint Hessian evaluations
        self.nohvpev = stats['Hprod']        # number of objective Hessian-vector product evaluations
        self.setup_time = stats['tsetup']   # CPU time for setup
        self.run_time = stats['trun']       # CPU time for run

        self.nfev = self.model_evals        # number of function evaluations
        self.ngev = self.deriv_evals        # number of derivative evaluations


def find_problems(objective=None, constraints=None, regular=None, degree=None, origin=None, internal=None, n=None, userN=None, m=None, userM=None):
    '''
    Returns a list of problem names for problems that satisfy the criteria provided.
    See CUTE classification scheme for a detailed description of the criteria.
    (http://www.cuter.rl.ac.uk/Problems/classification.shtml)

    This function just wraps the pycutest function `pycutest.find_problems()`.

    .. note::
        Problems with a user-settable number of variables/constraints will be match any given n / m.

    Warnings
    --------
        If a requirement is not provided, it is not applied. See below for details on the requirements.

    Parameters
    ----------
    objective: str
        A string containing one or more substrings 
        (``'none'``, ``'constant'``, ``'linear'``, ``'quadratic'``, ``'sum of squares'``, ``'other'``) 
        specifying the type of the objective function.
    constraints: str
        A string containing one or more substrings 
        (``'unconstrained'``, ``'fixed'``, ``'bound'``, ``'adjacency'``, ``'linear'``, ``'quadratic'``, ``'other'``) 
        specifying the type of the constraints.
    regular: bool
        ``True`` if the problem must be regular, ``False`` if it must be irregular.
    degree: list
        A list of the form ``[min, max]`` specifying the minimum and the maximum number of 
        analytically available derivatives.
    origin: str
        A string containing one or more substrings (``'academic'``, ``'modelling'``, ``'real-world'``) 
        specifying the origin of the problem.
    internal: bool
        A boolean, ``True`` if the problem must have internal variables, 
        ``False`` if internal variables are not allowed.
    n: list
        A list of the form ``[min, max]`` specifying the lowest and the highest allowed number of variables.
    userN: bool
        ``True`` if the problems must have user settable number of variables, 
        ``False`` if the number must be hardcoded.
    m: list
        A list of the form ``[min, max]`` specifying the lowest and the highest allowed number of constraints.
    userM: bool
        ``True`` if the problems must have user settable number of variables, 
        ``False`` if the number must be hardcoded.

    Returns
    -------
    prob_names: list
        A list of strings with problem names which satisfy the given requirements.
    
    Examples
    --------
    # Choose unconstrained, variable-dimension problems
    >>> probs = modopt.cutest.find_problems(constraints='unconstrained', userN=True)
    '''

    return pycutest.find_problems(objective=objective, constraints=constraints, regular=regular, degree=degree, 
                                  origin=origin, internal=internal, n=n, userN=userN, m=m, userM=userM)


def problem_properties(problem_name):
    '''
    Returns problem properties (based on the CUTEst problem classification string).
    See http://www.cuter.rl.ac.uk/Problems/classification.shtml for details on the properties.

    This function just wraps the pycutest function `pycutest.problem_properties()`.

    The output is a dictionary with the following members:

    * objective:    objective type (e.g., linear, quadratic, ...)
    * constraints:  constraints type (e.g., unconstrained, variable bounds only, ...)
    * regular:      ``True`` if the problem's first and second derivatives exist and are continuous everywhere
    * degree:       highest degree of analytically available derivatives (restricted to 0, 1, or 2)
    * origin:       problem origin (can be academic, modelling, or real-world problems)
    * internal:     ``True`` if problem has explicit internal variables
    * n:            number of variables (='variable' if it can be set by the user)
    * m:            number of constraints (='variable' if it can be set by the user)

    Parameters
    ----------
    problem_name: str
        Name of the CUTEst problem for which you need the problem properties.
    
    Returns
    -------
    problem_properties: dict
        A dictionary containing given problem's properties.
    
    Examples
    --------
    # Properties of problem ROSENBR
    >>> probs = modopt.cutest.problem_properties('ROSENBR')
    '''
    return pycutest.problem_properties(problem_name)

def print_available_sif_params(problem_name):
    '''
    This function calls sifdecode on a given problem to print out available optional input parameters.
    This function is OS dependent, and it currently works only on Linux and MacOS.

    This function just wraps the pycutest function `pycutest.print_available_sif_params()`.

    Parameters
    ----------
    problem_name: str
        Name of the CUTEst problem for which you need to print the sif parameters.
    
    Examples
    --------
    # Print optional sif parameters for problem ARGLALE
    >>> probs = modopt.cutest.print_available_sif_params('ARGLALE')
    '''
    pycutest.print_available_sif_params(problem_name)

def import_problem(problemName, destination=None, sifParams=None, sifOptions=None, efirst=False, lfirst=False, nvfirst=False, quiet=True, drop_fixed_variables=True):
    '''
    Prepares a problem interface module, imports and initializes it (all done by pycutest).

    This function wraps the pycutest function `pycutest.import_problem()`.

    Parameters
    ----------
    problemName: str
        CUTEst problem name.
    destination: str
        The name under which the compiled problem interface is stored in the cache (default = ``problemName``).
    sifParams: dict
        SIF file parameters to use (as dict, keys must be strings).
    sifOptions: list
        Additional options passed to sifdecode given in the form of a list of strings.
    efirst: bool
        Order equation constraints first (default ``True``).
    lfirst: bool
        Order linear constraints first (default ``True``).
    nvfirst: bool
        Order nonlinear variables before linear variables (default ``False``).
    quiet: bool
        Suppress output (default ``True``).
    drop_fixed_variables: bool
        in the resulting problem object, are fixed variables hidden from the user (default ``True``).

    Parameters
    ----------
    pycutest_problem: pycutest.CUTEstProblem
        A reference to the Python interface object for given problem_name, sifParams,
        and other arguments provided.
    
    Examples
    --------
    # Build the problem named ARGLALE with sif parameters N=100, M=200
    >>> problem = modopt.cutest.import_problem('ARGLALE', sifParams={'N':100, 'M':200})
    >>> print(problem)
    '''
    try: 
        return pycutest.import_problem(problemName, destination=destination, sifParams=sifParams, sifOptions=sifOptions, 
                                       efirst=efirst, lfirst=lfirst, nvfirst=nvfirst, quiet=quiet, 
                                       drop_fixed_variables=drop_fixed_variables)
    except RuntimeError as e:
        if 'SIFDECODE error:  Failed setting' in str(e):
            print_available_sif_params(problemName)
            raise RuntimeError(str(e) + ' for problem ' + problemName)
        else:
            raise RuntimeError(str(e))