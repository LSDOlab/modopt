import warnings
import numpy as np
import time

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not found, plotting disabled.")
    plt = None

def generate_performance_profiles(data):
    '''
    Generate the performance profiles for the given data.

    Parameters
    ----------
    data : dict
        Dictionary containing the performance data for each solver.
        The keys are the (problem_name, solver_name) and the values are 
        dictionaries containing 'time' and `success` as keys with corresponding
        values denoting the time (float) taken for 'solver_name' to solve 'problem_name' 
        and the success (bool) of the solver.
    Returns
    -------
    Tau : numpy.ndarray
        Array of log-scaled performance ratios.
    performance_profiles : dict
        Dictionary containing the performance profiles for each solver.
        The keys are the solver names and the values are the proportion of problems
        solved under the performance ratio corresponding to entries in Tau.
    '''
    if plt is None:
        raise ImportError("matplotlib not found, cannot plot performance profile.")
    
    # Get the unique solvers and problems
    solvers  = np.unique([key[1] for key in data.keys()])
    problems = np.unique([key[0] for key in data.keys()])

    # Get the minimum time taken by any solver for a given problem
    min_times = {}
    for problem in problems:
        min_times[problem] = np.min([data[(problem, solver)]['time'] for solver in solvers])

    # Compute the performance profile
    perf_ratio = {}
    for solver in solvers:
        for problem in problems:
            if data[(problem, solver)]['success']:
                perf_ratio[(problem, solver)] = data[(problem, solver)]['time'] / min_times[problem]
            else:
                perf_ratio[(problem, solver)] = np.inf

    # Get the maximum performance ratio over all problems
    max_perf_ratio = np.max(list(perf_ratio.values()))
    
    # Replace inf with 10 * max_perf_ratio
    # perf_ratio = {key: 10 * max_perf_ratio if value == np.inf else value for key, value in perf_ratio.items()}
    for key, value in perf_ratio.items():
        if value == np.inf:
            perf_ratio[key] = 10 * max_perf_ratio

    def performance_function(Tau):
        performance_profiles = {}
        for solver in solvers:
            performance_profiles[solver] = []
            for t in Tau:
                # Number of problems solved under tau performance ratio
                n_solved = np.sum([1 if np.log2(value) <= t else 0 for key, value in perf_ratio.items() if key[1] == solver])
                performance_profiles[solver].append(n_solved / len(problems))
        
        return performance_profiles
    
    Tau = np.linspace(0, np.log2(max_perf_ratio*10), 100)
    performance_profiles = performance_function(Tau)

    return Tau, performance_profiles
    
def plot_performance_profiles(data, save_figname='performance.pdf'):
    '''
    Plot the performance profile for the given data.

    Parameters
    ----------
    data : dict
        Dictionary containing the performance data for each solver.
        The keys are the (problem_name, solver_name) and the values are 
        dictionaries containing 'time' and `success` as keys with corresponding
        values denoting the time (float) taken for 'solver_name' to solve 'problem_name' 
        and the success (bool) of the solver.
    save_figname : str, optional
        Path to save the performance profile plot. Default is 'performance.pdf'.
    '''
    fig, ax = plt.subplots()
    ax.set_title('Performance Profile')
    ax.set_xlabel('Performance Ratio')
    ax.set_ylabel('Proportion of Problems')

    Tau, performance_profiles = generate_performance_profiles(data)

    for solver, profile in performance_profiles.items():
        ax.plot(Tau, profile, label=solver)

    ax.legend()
    fig.set_size_inches(10, 6)
    plt.savefig(save_figname)
    plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()