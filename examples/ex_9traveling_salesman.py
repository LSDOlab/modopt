'''Traveling Salesman Problem'''

import numpy as np
from modopt import Problem

class TravelingSalesman(Problem):
    def initialize(self, ):
        self.problem_name = 'traveling_salesman'
        # self.options.declare('locations', types=np.ndarray)

    def setup(self):
        self.add_design_variables('x',
                                  shape=(100, ),
                                  vals=np.arange(100, dtype=int))
        np.random.seed(0)
        # (x,y) locations of the 100 cities in a 100x100 square
        self.locations = np.random.random_sample((100,2)) * 100
        # self.locations = self.options['locations']
        self.add_objective('total_distance')

        # get_neighbor attribute of the problem is a function that generates a random neighbor
        self.get_neighbor = lambda x: self.generate_random_neighbor(x)

    def setup_derivatives(self):
        pass

    def compute_objective(self, dvs, obj):
        # print(self.locations)
        # print(dvs['x'].astype(int))
        ordered_city_locations = self.locations[dvs['x'].astype(int)]
        coordinate_differences = np.roll(ordered_city_locations, -1 , axis=0) - ordered_city_locations

        obj['total_distance'] = np.sum(np.linalg.norm(coordinate_differences, axis=1))

    def generate_random_neighbor(self, x):
        random_segment = np.random.choice(100, 2, replace=False) # returns float array
        # print(random_segment)
        i = np.min(random_segment)
        j = np.max(random_segment)

        neighbor = x * 1
        # reversed_segment = np.flip(x[i:j])
        neighbor[i:j] = np.flip(x[i:j])

        # print(neighbor)
        return neighbor


from modopt import SimulatedAnnealing

std_dev_tol = 1. # 1 unit of distance standard deviation for the last 1000 iterations
std_dev_sample_size = 1000
maxiter = 15000
T0 = 50.
settling_time = 100

# np.random.seed(0)
# prob = TravelingSalesman(locations=np.random.random_sample((100,2)) * 100)
prob = TravelingSalesman()

# Set up your optimizer with the problem
optimizer = SimulatedAnnealing(prob, 
                               maxiter=maxiter, 
                               std_dev_tol=std_dev_tol,
                               std_dev_sample_size=std_dev_sample_size,
                               T0=T0, 
                               settling_time=settling_time
                               )

optimizer.solve()
optimizer.print_results()

# import matplotlib.pyplot as plt
# y = np.loadtxt("traveling_salesman_outputs/obj.out")
# plt.plot(y)
# plt.show()

print('Simulated Annealing results:')

print('optimized_dvs:', optimizer.results['x'])
print('optimized_obj:', optimizer.results['f'])
print('initial_obj:', optimizer.results['f0'])
print('improvement from initial obj:', optimizer.results['f0'] - optimizer.results['f'], '[', optimizer.results['improvement'] * 100, '%]')
print('obj std dev for last 1000 iterations:', optimizer.results['f_sd'])
print('total number of moves:', optimizer.results['nmoves'])
print('total time taken:', optimizer.results['time'])
print('total number of iterations:', optimizer.results['niter'])
print('total number of function evaluations:', optimizer.results['nfev'])
print('converged:', optimizer.results['converged'])