'''Import and solve a CUTEst problem using modopt'''

import pycutest as pc

# Get a list of unconstrained problems with user-variable number of variables
prob_list = pc.find_problems(constraints='unconstrained', userN=True)
print(prob_list)

# Get the properties of the ROSENBR problem
props = pc.problem_properties('ROSENBR')
print(props)

# Print the available SIF parameters for the ROSENBR problem
pc.print_available_sif_params('ROSENBR')

# Import the ROSENBR problem with no SIF parameters
pycutest_prob = pc.import_problem('ROSENBR', sifParams={})

if __name__ == "__main__":
    from modopt import CUTEstProblem, SLSQP

    print('Solve the second-order unconstrained problem: ROSENBR')
    print('-'*60)

    # Wrap the PyCUTEst problem for modOpt use with the CUTEstProblem class
    prob = CUTEstProblem(cutest_problem=pc.import_problem('ROSENBR'))

    # Solve the problem using the SLSQP optimizer
    optimizer = SLSQP(prob, solver_options={'maxiter':100})
    optimizer.solve()

    print(optimizer.results)
    print('-'*60)