import warnings
import numpy as np
import time

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not found, plotting disabled.")
    plt = None

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_theme(rc={'text.usetex': True})
# sns.set_style("ticks")

def generate_performance_profiles(data):
    '''
    Compute performance profiles and return them along 
    with their corresponding performance ratio (`Tau`) values.

    Depending on the input data, the function returns either two or four outputs:
    
    - If `'nev'` exists in `list(data.values())[0]`:

      - `Tau` (np.ndarray): Array of `Tau` values for the primary (time) performance profiles.

      - `performance_profiles` (np.ndarray): Performance profiles corresponding to `Tau`.

      - `Tau_n` (np.ndarray): Array of `Tau` values for secondary (nev) performance profiles.

      - `performance_profiles_n` (np.ndarray): Performance profiles corresponding to `Tau_n`.

    - Otherwise:

      - `Tau` (np.ndarray): Array of `Tau` values for the primary (time) performance profiles.

      - `performance_profiles` (np.ndarray): Performance profiles corresponding to `Tau`.

    Parameters
    ----------
    data : dict
        Dictionary containing the performance data for each solver.
        The keys are the (problem_name: str, solver_name: str) and the values are 
        dictionaries containing `'time'` and `'success'` as keys with corresponding
        values denoting the time (`float`) taken for `solver_name` to solve `problem_name`
        and the success (`bool`) of the solver.
        Additionally, if the number of evaluations is available, 
        the dictionary can also contain `'nev'` as a key with 
        the corresponding `int` value denoting the number of evaluations.

    Returns
    -------
    Tau : numpy.ndarray
        Array of log-scaled performance ratios.
    performance_profiles : dict
        Dictionary containing the performance profiles for each solver.
        The keys are the solver names and the values are the proportion of problems
        solved under the performance ratio corresponding to entries in Tau.
    Tau_n : numpy.ndarray
        Array of log-scaled performance ratios for the number of evaluations.
        Only returned if the number of evaluations `'nev'` is available in the data.
    performance_profiles_n : dict
        Dictionary containing the performance profiles for the number of evaluations.
        The keys are the solver names and the values are the proportion of problems
        solved under the performance ratio corresponding to entries in Tau_n.
        Only returned if the number of evaluations `'nev'` is available in the data.
    '''
    
    # Get the unique solvers and problems
    solvers  = np.unique([key[1] for key in data.keys()])
    problems = np.unique([key[0] for key in data.keys()])

    # Get the minimum time taken by any solver for a given problem
    min_times = {}
    for problem in problems:
        successful_times = [data[(problem, solver)]['time'] for solver in solvers if data[(problem, solver)]['success']]
        if successful_times == []:
            min_times[problem] = 1.0 # Put any non-zero value since all solvers will be using a large time
        else:
            min_times[problem] = np.min(successful_times)

        if min_times[problem] == 0:
            raise ValueError('Time taken by a successful solver for problem {} is 0.'.format(problem))

    # Compute the performance ratio - time
    perf_ratio = {}
    for solver in solvers:
        for problem in problems:
            if data[(problem, solver)]['success']:
                perf_ratio[(problem, solver)] = data[(problem, solver)]['time'] / min_times[problem]
            else:
                perf_ratio[(problem, solver)] = np.inf

    # Get the maximum performance ratio over all problems
    successful_perf_ratios = [value for value in perf_ratio.values() if value != np.inf]
    if successful_perf_ratios == []:
        raise ValueError('All solvers failed on all problems.')
    max_perf_ratio = np.max(successful_perf_ratios)
    
    # Replace inf with 10 * max_perf_ratio
    # perf_ratio = {key: 10 * max_perf_ratio if value == np.inf else value for key, value in perf_ratio.items()}
    for key, value in perf_ratio.items():
        if value == np.inf:
            perf_ratio[key] = 10 * max_perf_ratio

    # Compute the performance ratio - number of evaluations
    if 'nev' in data[(problems[0], solvers[0])]:
        min_nevs = {}
        for problem in problems:
            successful_nevs = [data[(problem, solver)]['nev'] for solver in solvers if data[(problem, solver)]['success']]
            if successful_nevs == []:
                min_nevs[problem] = 1 # Put any non-zero value since all solvers will be using a large nev
            else:
                min_nevs[problem] = np.min(successful_nevs)

            if min_nevs[problem] == 0:
                raise ValueError('Number of evaluations by a successful solver for problem {} is 0.'.format(problem))

        perf_ratio_n = {}
        for solver in solvers:
            for problem in problems:
                if data[(problem, solver)]['success']:
                    perf_ratio_n[(problem, solver)] = data[(problem, solver)]['nev'] / min_nevs[problem]
                else:
                    perf_ratio_n[(problem, solver)] = np.inf

        # The following block is redundant since this is already done for time
        # successful_perf_ratios_n = [value for value in perf_ratio_n.values() if value != np.inf]
        # if successful_perf_ratios_n == []:
        #     raise ValueError('All solvers failed on all problems.')
        
        max_perf_ratio_n = np.max([value for value in perf_ratio_n.values() if value != np.inf])

        # Replace inf with 10 * max_perf_ratio_n
        for key, value in perf_ratio_n.items():
            if value == np.inf:
                perf_ratio_n[key] = 10 * max_perf_ratio_n

    def performance_function(Tau, perf_ratio):
        performance_profiles = {}
        for solver in solvers:
            performance_profiles[solver] = []
            for t in Tau:
                # Number of problems solved under tau performance ratio
                n_solved = np.sum([1 if np.log2(value) <= t else 0 for key, value in perf_ratio.items() if key[1] == solver])
                performance_profiles[solver].append(n_solved / len(problems))
        
        return performance_profiles
    
    Tau = np.linspace(0, np.log2(max_perf_ratio*10), 100)[:-1] # upper bound 10*max_perf_ratio needs to be omitted
    performance_profiles = performance_function(Tau, perf_ratio)

    print('Total number of problems:', len(problems), '\n')
    for solver in solvers:
        print('Solver:', solver)
        print('-'*50)
        print('Number of problems solved:', int(np.ceil(performance_profiles[solver][-2]*len(problems))))
        print('Percentage of problems solved:', performance_profiles[solver][-2]*100)
        print('-'*50, '\n')

    if 'nev' in data[(problems[0], solvers[0])]:
        Tau_n = np.linspace(0, np.log2(max_perf_ratio_n*10), 100)[:-1] # upper bound 10*max_perf_ratio_n needs to be omitted
        performance_profiles_n = performance_function(Tau_n, perf_ratio_n)

        return Tau, performance_profiles, Tau_n, performance_profiles_n

    return Tau, performance_profiles
    
def plot_performance_profiles(data, save_figname='performance.pdf'):
    '''
    Plot the performance profiles for the given data.

    Parameters
    ----------
    data : dict
        Dictionary containing the performance data for each solver.
        The keys are the (problem_name: str, solver_name: str) and the values are 
        dictionaries containing `'time'` and `'success'` as keys with corresponding
        values denoting the time (`float`) taken for `solver_name` to solve `problem_name`
        and the success (`bool`) of the solver.
        Additionally, if the number of evaluations is available, 
        the dictionary can also contain `'nev'` as a key with 
        the corresponding `int` value denoting the number of evaluations.
    save_figname : str, default='performance.pdf'
        Path to save the plot with the performance profiles.
        If the number of evaluations is available in `data`, 
        the performance profiles for the number of evaluations is also plotted
        and saved to the path with `save_figname` appended with `_nev` before the extension.
    '''
    
    if plt is None:
        raise ImportError("matplotlib not found, cannot plot performance profile.")
    
    plt.rcParams['xtick.labelsize']=20
    plt.rcParams['ytick.labelsize']=20
    
    fig, ax = plt.subplots()
    ax.set_title('Performance Profile (time)', fontsize=24)
    ax.set_xlabel('Logarithmic performance ratio, $log_2(\\tau)$', fontsize=24)
    ax.set_ylabel('Proportion of problems solved', fontsize=24)

    if 'nev' not in data[(list(data.keys())[0][0], list(data.keys())[0][1])]:
        Tau, performance_profiles = generate_performance_profiles(data)
    
    else:
        Tau, performance_profiles, Tau_n, performance_profiles_n = generate_performance_profiles(data)

    for solver, profile in performance_profiles.items():
        ax.plot(Tau, profile, label=solver, linewidth = 2.0)

    ax.legend(fontsize=18)
    ax.set_xlim([0., Tau[-1]])
    ax.set_ylim([0., 1.])
    plt.minorticks_off()
    fig.set_size_inches(8, 6)
    plt.savefig(save_figname, bbox_inches='tight')
    plt.show()

    if 'nev' in data[(list(data.keys())[0][0], list(data.keys())[0][1])]:
        fig, ax = plt.subplots()
        ax.set_title('Data Profile (function evaluations)', fontsize=24)
        ax.set_xlabel('Logarithmic performance ratio, $log_2(\\tau)$', fontsize=24)
        ax.set_ylabel('Proportion of problems solved', fontsize=24)

        for solver, profile in performance_profiles_n.items():
            ax.plot(Tau_n, profile, label=solver, linewidth = 2.0)

        ax.legend(fontsize=18)
        ax.set_xlim([0., Tau_n[-1]])
        ax.set_ylim([0., 1.])
        plt.minorticks_off()
        fig.set_size_inches(8, 6)
        plt.savefig(save_figname.replace('.pdf', '_nev.pdf'), bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()