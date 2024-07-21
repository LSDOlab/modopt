import warnings
import numpy as np
try:
    import h5py
except ImportError:
    warnings.warn("h5py not found, saving and loading data disabled")
    h5py = None

def import_h5py_file(filepath):
    if h5py is None:
        raise ImportError("h5py not found, saving and loading data disabled")
    
    return h5py.File(filepath, 'r')
    
def print_record_contents(filepath, suppress_print=False):
    '''
    Print and return the contents of the record file.
    
    Parameters
    ----------
    filepath : str
        Path to the record file.
    suppress_print : bool, default=False
        If False, print the contents of the record file.
        Otherwise, return the contents as a tuple.

    Returns
    -------
    attributes : list
        List of attributes of optimization.
    opt_vars : list
        List of recorded optimizer variables.
    callback_vars : list
        List of recorded callback variables.
    results : list
        List of results of optimization.

    Examples
    --------
    >>> import numpy as np
    >>> import modopt as mo
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> con = lambda x: np.array([x[0] + x[1], x[0] - x[1]])
    >>> jac = lambda x: np.array([[1, 1], [1, -1]])
    >>> xl = np.array([1.0, -np.inf])
    >>> x0 = np.array([500., 50.])
    >>> cl = 1.0
    >>> cu = np.array([1., np.inf])
    >>> problem = mo.ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, xl=xl, cl=cl, cu=cu)
    >>> optimizer = mo.SLSQP(problem, recording=True)
    >>> results   = optimizer.solve()
    >>> from modopt.postprocessing import print_record_contents
    >>> print_record_contents(optimizer.out_dir+'/record.hdf5')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Available data in the record:
    -----------------------------
     - Attributes of optimization    : ['c_lower', 'c_scaler', 'c_upper', 'constrained', 'hot_start_from', 'modopt_output_files', 'nc', 
       'nx', 'o_scaler', 'problem_name', 'readable_outputs', 'recording', 'solver_name', 'solver_options-callback', 'solver_options-disp', 
       'solver_options-ftol', 'solver_options-maxiter', 'timestamp', 'visualize', 'x0', 'x_lower', 'x_scaler', 'x_upper']
     - Recorded optimizer variables  : ['x']
     - Recorded callback variables   : ['con', 'grad', 'jac', 'obj', 'x']
     - Results of optimization       : ['fun', 'jac', 'message', 'nfev', 'nit', 'njev', 'out_dir', 'status', 'success', 'total_callbacks', 'x']
    ([...], [...], [...])
    '''
    file = import_h5py_file(filepath)

    attributes = list(file.attrs.keys())

    if 'iteration_0' in file.keys():
        opt_vars = list(file['iteration_0'].keys())
    else:
        opt_vars = []

    callback_vars = set()
    for key in file.keys():
        if key.startswith('callback_'):
            in_vars  = set(file[key]['inputs'].keys())
            out_vars = set(file[key]['outputs'].keys())
            callback_vars = callback_vars | in_vars | out_vars
    callback_vars  = sorted(list(callback_vars))

    try:
        results = list(file['results'].keys())
    except:
        results = []
        warnings.warn("No results found in the record file.")

    file.close()

    if not suppress_print:
        print("Available data in the record:")
        print("-----------------------------")
        print(f" - {'Attributes of optimization':30}:", attributes)
        print(f" - {'Recorded optimizer variables':30}:", opt_vars)
        print(f" - {'Recorded callback variables':30}:", callback_vars)
        print(f" - {'Results of optimization':30}:", results)

    return attributes, opt_vars, callback_vars, results

def load_results(filepath):
    '''
    Load the results of optimization from the record as a dictionary.

    Parameters
    ----------
    filepath : str
        Path to the record file.

    Returns
    -------
    out_data : dict
        Dictionary with optimization results.

    Examples
    --------
    >>> import numpy as np
    >>> import modopt as mo
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> con = lambda x: np.array([x[0] + x[1], x[0] - x[1]])
    >>> jac = lambda x: np.array([[1, 1], [1, -1]])
    >>> xl = np.array([1.0, -np.inf])
    >>> x0 = np.array([500., 50.])
    >>> cl = 1.0
    >>> cu = np.array([1., np.inf])
    >>> problem = mo.ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, xl=xl, cl=cl, cu=cu)
    >>> optimizer = mo.SLSQP(problem, recording=True)
    >>> results   = optimizer.solve()
    >>> from modopt.postprocessing import load_results
    >>> load_results(optimizer.out_dir+'/record.hdf5')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'fun': 1.0000000209233804, 'jac': array([ 2.00000002e+00, -2.09231530e-08]), 
    'message': 'Optimization terminated successfully', 'nfev': 2, 'nit': 2, 'njev': 2, 'out_dir': '...',
    'status': 0, 'success': True, 'total_callbacks': 9, 'x': array([ 1.0000..., -1.0...e-08])}

    '''
    file = import_h5py_file(filepath)
    result_dict = {}
    for key in file['results'].keys():
        result_dict[key] = file['results'][key][()]
        if key in ['message', 'out_dir']:
            result_dict[key] = result_dict[key].decode('utf-8')
    file.close()
    return result_dict

def load_attributes(filepath):
    '''
    Load the attributes of optimization from the record as a dictionary.

    Parameters
    ----------
    filepath : str
        Path to the record file.

    Returns
    -------
    out_data : dict
        Dictionary with optimization attributes.

    Examples
    --------
    >>> import numpy as np
    >>> import modopt as mo
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> con = lambda x: np.array([x[0] + x[1], x[0] - x[1]])
    >>> jac = lambda x: np.array([[1, 1], [1, -1]])
    >>> xl = np.array([1.0, -np.inf])
    >>> x0 = np.array([500., 50.])
    >>> cl = 1.0
    >>> cu = np.array([1., np.inf])
    >>> problem = mo.ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, xl=xl, cl=cl, cu=cu)
    >>> optimizer = mo.SLSQP(problem, recording=True)
    >>> results   = optimizer.solve()
    >>> from modopt.postprocessing import load_attributes
    >>> load_attributes(optimizer.out_dir+'/record.hdf5')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'c_lower': array([1., 1.]), 'c_scaler': array([1., 1.]), 'c_upper': array([ 1., inf]), 'constrained': True, 
    'hot_start_from': 'None', 'modopt_output_files': ['directory: ...', 'modopt_results.out', 'record.hdf5'], 
    'nc': 2, 'nx': 2, 'o_scaler': array([1.]), 'problem_name': 'unnamed_problem', 'readable_outputs': [], 
    'recording': 'True', 'solver_name': 'scipy-slsqp', 'solver_options-callback': 'None', 'solver_options-disp': False, 
    'solver_options-ftol': 1e-06, 'solver_options-maxiter': 100, 'timestamp': '...', 'visualize': [], 
    'x0': array([500.,  50.]), 'x_lower': array([  1., -inf]), 'x_scaler': array([1., 1.]), 'x_upper': array([inf, inf])}
    '''
    file = import_h5py_file(filepath)
    attr_dict = {}
    for key in file.attrs.keys():
        attr_dict[key] = file.attrs[key]
        if key in ['visualize', 'modopt_output_files', 'readable_outputs']:
            attr_dict[key] = list(file.attrs[key])
    file.close()
    return attr_dict


def load_variables(filepath, vars):
    '''
    Load specified scalar variable iterates from the record file.
    Returns a dictionary with the variable names as keys and list of variable iterates as values.
    Note that the keys for callback variables will be prefixed with 'callback_'
    as opposed to optimizer variables that will have same key as the specified variable name.

    Parameters
    ----------
    filepath : str
        Path to the record file.
    vars : str or list
        Variable names to load from the record file.

    Returns
    -------
    out_data : dict
        Dictionary with variable names as keys and list of variable iterates as values.
        Keys for callback variables will be prefixed with 'callback_'.

    Examples
    --------
    >>> import numpy as np
    >>> import modopt as mo
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> con = lambda x: np.array([x[0] + x[1], x[0] - x[1]])
    >>> jac = lambda x: np.array([[1, 1], [1, -1]])
    >>> xl = np.array([1.0, -np.inf])
    >>> x0 = np.array([500., 50.])
    >>> cl = 1.0
    >>> cu = np.array([1., np.inf])
    >>> problem = mo.ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, xl=xl, cl=cl, cu=cu)
    >>> optimizer = mo.SLSQP(problem, recording=True)
    >>> results   = optimizer.solve()
    >>> from modopt.postprocessing import load_variables
    >>> load_variables(optimizer.out_dir+'/record.hdf5', ['x[0]', 'obj', 'con[1]', 'grad[0]', 'jac[0,1]']) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'x[0]': [500.0, 1.0000000..., 1.00000...], 
    'callback_x[0]': [500.0, 500.0, 500.0, 500.0, 500.0, 1.00000..., 1.000000..., 1.00000..., 1.000000...], 
    'callback_obj': [252500.0, 1.000000...], 'callback_con[1]': [450.0, 450.0, 1.000000...], 
    'callback_grad[0]': [1000.0, 2.0000000...], 'callback_jac[0,1]': [1.0, 1.0]}

    '''
    if not isinstance(filepath, str):
        raise ValueError("'filepath' must be a string.")
    if not isinstance(vars, (str, list)):
        raise ValueError("'vars' must be a string or a list of strings")
    if isinstance(vars, str):
        vars = [vars]
    if not all(isinstance(var, str) for var in vars):
        raise ValueError("'vars' must be a string or a list of strings")
    
    attrs, opt_vars, callback_vars, results = print_record_contents(filepath, suppress_print=True)
    
    file  = import_h5py_file(filepath)
    n_iter = len([key for key in file.keys() if key.startswith('iteration_')])
    n_cb   = len([key for key in file.keys() if key.startswith('callback_')])

    out_data = {}
    for s_var in vars:
        var = s_var.split('[')[0]
        if var not in opt_vars+callback_vars:
            raise ValueError(f"Variable {var} not found in any of the callbacks or optimizer output data in the record.")
        if var in opt_vars:
            out_data[s_var] = []
        if var in callback_vars:
            out_data[f'callback_{s_var}'] = []

    for i in range(n_iter):
        for s_var in vars:
            var = s_var.split('[')[0]
            if var not in opt_vars:
                continue
            if '[' not in s_var:
                out_data[s_var].append(file[f'iteration_{i}'][var][()])
            elif ',' not in s_var:
                idx = int(s_var.split('[')[1].split(']')[0])
                out_data[s_var].append(file[f'iteration_{i}'][var][idx])
            else:
                idx1, idx2 = map(int, s_var.split('[')[1].split(']')[0].split(','))
                out_data[s_var].append(file[f'iteration_{i}'][var][idx1, idx2])

    for i in range(n_cb):
        for s_var in vars:
            var = s_var.split('[')[0]

            if var not in callback_vars:
                continue

            current_cb_vars = list(file[f'callback_{i}']['outputs'].keys()) + list(file[f'callback_{i}']['inputs'].keys())
            if var not in current_cb_vars:
                continue
            
            group_key = 'outputs' if var in list(file[f'callback_{i}']['outputs'].keys()) else 'inputs'
            if '[' not in s_var:
                out_data[f'callback_{s_var}'].append(file[f'callback_{i}'][group_key][var][()])
            elif ',' not in s_var:
                idx = int(s_var.split('[')[1].split(']')[0])
                out_data[f'callback_{s_var}'].append(file[f'callback_{i}'][group_key][var][idx])
            else:
                idx1, idx2 = map(int, s_var.split('[')[1].split(']')[0].split(','))
                out_data[f'callback_{s_var}'].append(file[f'callback_{i}'][group_key][var][idx1, idx2])
    
    file.close()

    return out_data

def print_dict_as_table(data):
    """
    Print any input dictionary as a table.

    Parameters
    ----------
    data : dict
        Dictionary to print as a table.

    Examples
    --------
    >>> data = {'a': 0, 'b': "string", 'c': ['a', 'b', 'c']}
    >>> print_dict_as_table(data)
    --------------------------------------------------
            a                        : 0
            b                        : string
            c                        : ['a', 'b', 'c']
    --------------------------------------------------
    """
    print("--------------------------------------------------")
    for key, value in data.items():
        print(f"        {key:24} : {value}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    import doctest
    doctest.testmod()