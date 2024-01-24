'''Example 8 : Traveling Salesman Problem'''
import numpy as np
from modopt.api import Problem

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


    def setup_derivatives(self):
        pass

    def compute_objective(self, dvs, obj):
        # print(self.locations)
        # print(dvs['x'].astype(int))
        ordered_city_locations = self.locations[dvs['x'].astype(int)]
        coordinate_differences = np.roll(ordered_city_locations, -1 , axis=0) - ordered_city_locations

        obj['total_distance'] = np.sum(np.linalg.norm(coordinate_differences, axis=1))

    def generate_random_neighbor(self, dvs):
        random_segment = np.random.choice(100, 2, replace=False) # returns float array
        # print(random_segment)
        i = np.min(random_segment)
        j = np.max(random_segment)

        neighbor = dvs * 1
        # reversed_segment = np.flip(dvs[i:j])
        neighbor[i:j] = np.flip(dvs[i:j])

        # print(neighbor)

        # breakpoint()

        return neighbor




from modopt.optimization_algorithms import SimulatedAnnealing

tol = 1.
max_itr = 10000
T0 = 50.
settling_time = 100

# np.random.seed(0)
# prob = TravelingSalesman(locations=np.random.random_sample((100,2)) * 100)
prob = TravelingSalesman()

# Set up your optimizer with the problem
optimizer = SimulatedAnnealing(prob, max_itr=max_itr, tol=tol, T0=T0, settling_time=settling_time)

optimizer.solve()
optimizer.print_results(summary_table=True)

# import matplotlib.pyplot as plt
# y = np.loadtxt("traveling_salesman_outputs/obj.out")
# plt.plot(y)
# plt.show()

print('Simulated Annealing results:')
print('optimized_dvs:', prob.x.get_data())
print('optimized_obj:', prob.obj['total_distance'])