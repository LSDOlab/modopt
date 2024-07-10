from modopt.external_libraries.scipy import SLSQP, COBYLA, BFGS, LBFGSB, NelderMead, COBYQA, TrustConstr
from modopt.external_libraries.pyslsqp import PySLSQP
from modopt.external_libraries.snopt import SNOPTc as SNOPT
from modopt.external_libraries.ipopt import IPOPT
from modopt.external_libraries.cvxopt import CVXOPT
from modopt.external_libraries.qpsolvers import ConvexQPSolvers

def optimize(prob, solver='SLSQP', **kwargs):
    """
    Optimize a given problem using a specified solver.

    Only performant algorithms can be used with this function
    and is made available for users who are only interested in
    solving their optimization problems with solvers in the modOpt library.
    Developers of instructional algorithms are recommended to use
    `Optimizer` subclasses such as SLSQP, Newton, NelderMead, etc. directly.

    Parameters
    ----------
    prob : Problem or ProblemLite
        The problem to be solved.
    solver : str, optional
        The solver to be used. Default is 'SLSQP'.
        Available solvers are 'SLSQP', 'PySLSQP', 'COBYLA', 'BFGS',
        'LBFGSB', 
        'SNOPT', 'IPOPT', 'CVXOPT', and 'ConvexQPSolvers'.
    **kwargs
        Additional keyword arguments to be passed to the solver.

    Returns
    -------
    dict
        The results of the optimization.
    """
    valid_solvers = ['SLSQP', 'PySLSQP', 'COBYLA', 'BFGS', 
                     'LBFGSB',
                     'SNOPT', 'IPOPT', 'CVXOPT', 'ConvexQPSolvers']
    if solver == 'SLSQP':
        optimizer = SLSQP(prob, **kwargs)
    elif solver == 'PySLSQP':
        optimizer = PySLSQP(prob, **kwargs)
    elif solver == 'COBYLA':
        optimizer = COBYLA(prob, **kwargs)
    elif solver == 'BFGS':
        optimizer = BFGS(prob, **kwargs)
    elif solver == 'LBFGSB':
        optimizer = LBFGSB(prob, **kwargs)
    elif solver == 'SNOPT':
        optimizer = SNOPT(prob, **kwargs)
    elif solver == 'IPOPT':
        optimizer = IPOPT(prob, **kwargs)
    elif solver == 'CVXOPT':
        optimizer = CVXOPT(prob, **kwargs)
    elif solver == 'ConvexQPSolvers':
        optimizer = ConvexQPSolvers(prob, **kwargs)
    else:
        raise ValueError(f"Invalid solver named '{solver}' is specified. Valid solvers are: {valid_solvers}.")

    return optimizer.solve()