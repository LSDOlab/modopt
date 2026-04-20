from modopt import CUTEstProblem
import pycutest

from modopt.benchmarking import filter_cutest_problems

range_vars = [1,10000000000]
range_cons = [0,10000000000]

# Check FP tag
##############
counter = 0
prob_list = filter_cutest_problems(num_vars=range_vars, num_cons=range_cons, tags=['FP'])

# Loop over all selected problems, checking if they match the tags
for i, prob_name in enumerate(prob_list):

    obj_type = pycutest.problem_properties(prob_name)['objective']

    if obj_type not in ['none', 'constant']:
        print(f"Problem {prob_name} has objective function, but is tagged as FP." \
              +f"Please check the tags for this problem.")
        counter += 1
        
print('Finished checking FP tags. Failed checks:', counter)

# Check LP tag
##############
counter = 0
prob_list = filter_cutest_problems(num_vars=range_vars, num_cons=range_cons, tags=['LP'])

# Loop over all selected problems, checking if they match the tags
for i, prob_name in enumerate(prob_list):

    obj_type = pycutest.problem_properties(prob_name)['objective']
    con_type = pycutest.problem_properties(prob_name)['constraints']

    if obj_type not in ['linear'] or con_type not in ['linear', 'bound']:
        print(f"Problem {prob_name} is tagged as LP." \
              +f"Please check the tags for this problem.")
        counter += 1
        
print('Finished checking LP tags. Failed checks:', counter)

# Check QP tag
##############
counter = 0
prob_list = filter_cutest_problems(num_vars=range_vars, num_cons=range_cons, tags=['QP'])

# Loop over all selected problems, checking if they match the tags
for i, prob_name in enumerate(prob_list):

    obj_type = pycutest.problem_properties(prob_name)['objective']
    con_type = pycutest.problem_properties(prob_name)['constraints']

    if obj_type not in ['quadratic'] or \
       con_type not in ['unconstrained', 'fixed', 'linear', 'bound']:
        print(f"Problem {prob_name} is tagged as QP." \
              +f"Please check the tags for this problem.")
        counter += 1
        
print('Finished checking QP tags. Failed checks:', counter)
        
# Check UC tag
##############
counter = 0
prob_list = filter_cutest_problems(num_vars=range_vars, num_cons=range_cons, tags=['UC'])

# Loop over all selected problems, checking if they match the tags
for i, prob_name in enumerate(prob_list):

    con_type = pycutest.problem_properties(prob_name)['constraints']

    if con_type not in ['unconstrained', 'fixed']:
        print(f"Problem {prob_name} is tagged as UC." \
              +f"Please check the tags for this problem.")
        counter += 1

print('Finished checking UC tags. Failed checks:', counter)

# Check BC tag
##############
counter = 0
prob_list = filter_cutest_problems(num_vars=range_vars, num_cons=range_cons, tags=['BC'])

# Loop over all selected problems, checking if they match the tags
for i, prob_name in enumerate(prob_list):

    con_type = pycutest.problem_properties(prob_name)['constraints']

    if con_type not in ['bound']:
        print(f"Problem {prob_name} is tagged as BC." \
              +f"Please check the tags for this problem.")
        counter += 1

print('Finished checking BC tags. Failed checks:', counter)

# Check UC tag - more thorough and expensive with importing the problem
#######################################################################
counter = 0
import numpy as np

range_vars = [1,1000]
range_cons = [0,1000]
prob_list = filter_cutest_problems(num_vars=range_vars, num_cons=range_cons, tags=['UC'])

# Loop over all selected problems, checking if they match the tags
for i, prob_name in enumerate(prob_list):

    # Import pycutest problem
    pc_prob = pycutest.import_problem(prob_name)
    # print(f'[{i}.]', 'Problem name [num_vars, num_cons]:', prob_name, f'[{pc_prob.n}]', f'[{pc_prob.m}]')

    if pc_prob.m != 0:
        print(f"Problem {prob_name} is tagged as UC, but has {pc_prob.m} constraints." \
              +f"Please check the tags for this problem.")
        counter += 1

    # Create modopt problem
    prob = CUTEstProblem(cutest_problem=pc_prob)

    if prob.c_lower is not None or prob.c_upper is not None:
        print(f"Problem {prob_name} is tagged as UC, but has defined constraint bounds." \
              +f"Please check the tags for this problem.")
        counter += 1
    if np.any(prob.x_lower!=-np.inf) or np.any(prob.x_upper!=np.inf):
        print(f"Problem {prob_name} is tagged as UC, but has non-infinite variable bounds." \
              +f"Please check the tags for this problem.")
        counter += 1

print('Finished thorough check on UC tags. Failed checks:', counter)

# Check BC tag - more thorough and expensive with importing the problem
#######################################################################
counter = 0
import numpy as np

range_vars = [1,1000]
range_cons = [0,1000]
prob_list = filter_cutest_problems(num_vars=range_vars, num_cons=range_cons, tags=['BC'])

# Loop over all selected problems, checking if they match the tags
for i, prob_name in enumerate(prob_list):
        
    # Import pycutest problem
    pc_prob = pycutest.import_problem(prob_name)
    # print(f'[{i}.]', 'Problem name [num_vars, num_cons]:', prob_name, f'[{pc_prob.n}]', f'[{pc_prob.m}]')

    if pc_prob.m != 0:
        print(f"Problem {prob_name} is tagged as BC, but has {pc_prob.m} constraints." \
              +f"Please check the tags for this problem.")
        counter += 1

    # Create modopt problem
    prob = CUTEstProblem(cutest_problem=pc_prob)

    if prob.c_lower is not None or prob.c_upper is not None:
        print(f"Problem {prob_name} is tagged as BC, but has defined constraint bounds." \
              +f"Please check the tags for this problem.")
        counter += 1

    if np.all(prob.x_lower==-np.inf) and np.all(prob.x_upper==np.inf):
        print(f"Problem {prob_name} is tagged as BC, but has no variable bounds." \
              +f"Please check the tags for this problem.")
        print(prob.x_lower, prob.x_upper)
        counter += 1

print('Finished thorough check on BC tags. Failed checks:', counter)