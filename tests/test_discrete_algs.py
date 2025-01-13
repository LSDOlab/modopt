import numpy as np
from modopt import Problem, ProblemLite
import pytest

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

        self.get_neighbor = lambda x: self.generate_random_neighbor(x)

    def setup_derivatives(self):
        pass

    def compute_objective(self, dvs, obj):
        ordered_city_locations = self.locations[dvs['x'].astype(int)]
        coordinate_differences = np.roll(ordered_city_locations, -1 , axis=0) - ordered_city_locations

        obj['total_distance'] = np.sum(np.linalg.norm(coordinate_differences, axis=1))

    def generate_random_neighbor(self, x):
        random_segment = np.random.choice(100, 2, replace=False) # returns float array
        i = np.min(random_segment)
        j = np.max(random_segment)

        neighbor = x * 1
        # reversed_segment = np.flip(x[i:j])
        neighbor[i:j] = np.flip(x[i:j])

        return neighbor

def traveling_salesman_lite():
    np.random.seed(0)
    locations = np.random.random_sample((100,2)) * 100
    x0 = np.arange(100, dtype=int)
    
    obj = lambda x: np.sum(np.linalg.norm(np.roll(locations[x.astype(int)], -1 , axis=0) - locations[x.astype(int)], axis=1))

    def generate_random_neighbor(x):
        random_segment = np.random.choice(100, 2, replace=False) # returns float array
        i = np.min(random_segment)
        j = np.max(random_segment)

        neighbor = x * 1
        # reversed_segment = np.flip(x[i:j])
        neighbor[i:j] = np.flip(x[i:j])

        return neighbor
    
    prob_lite = ProblemLite(x0, obj=obj, name='traveling_salesman_lite', grad_free=True)
    prob_lite.get_neighbor = generate_random_neighbor

    return prob_lite

@pytest.mark.slow
def test_simulated_annealing():
    '''
    Test the Simulated Annealing algorithm for discrete, unconstrained problems.
    '''
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
    from modopt import SimulatedAnnealing

    prob = TravelingSalesman()
    solver_options = {
        'maxiter': 15000,
        'std_dev_tol': 1.,
        'std_dev_sample_size': 1000,
        'T0': 50.,
        'settling_time': 100
        }
    optimizer = SimulatedAnnealing(prob, **solver_options)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results()

    assert_array_equal(optimizer.results['x'], [
        83, 63, 66,  7, 86, 30, 23, 27, 47, 38, 39, 79, 85, 43, 95, 24, 98, 
        32, 40, 29, 42, 91, 31, 28, 64, 50, 56, 94, 84, 21, 48, 89, 37, 49, 
        33, 34, 26, 87, 70,  4, 35,  5, 57, 80, 69, 65, 25, 14, 20, 52, 76, 
        90, 45, 62, 16,  2,  0, 68, 22, 18, 93, 67, 53,  1, 60, 97, 78, 58, 
        92, 36, 61, 19, 59, 72, 82, 10,  9, 81, 51, 74, 44,  6, 71,  3, 88, 
        55, 11, 15, 54, 77, 75, 96, 13, 73,  8, 46, 12, 41, 17, 99
        ])
    assert_almost_equal(optimizer.results['f0'], 4576.3359328082515, decimal=11)
    assert_almost_equal(optimizer.results['objective'], 961.3934251787592, decimal=11)
    assert optimizer.results['converged']
    assert optimizer.results['niter'] <= solver_options['maxiter']
    assert optimizer.results['niter'] == 11394
    assert optimizer.results['nfev'] == 11395
    assert optimizer.results['nmoves'] == 1156
    assert_almost_equal(optimizer.results['improvement'], 0.789920705277245, decimal=11)


    prob = traveling_salesman_lite()
    optimizer = SimulatedAnnealing(prob, **solver_options)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results()

    assert_array_equal(optimizer.results['x'], [
        17, 41, 46,  8, 77, 75, 73, 13, 96, 54, 88, 71,  3, 44,  6, 55, 11,
         0, 93,  1, 53, 67, 68, 18, 22, 59, 72,  9, 81, 51, 74, 82, 10, 19,
        61, 92, 36, 35, 58, 78,  5, 97, 60, 80, 69, 65, 64, 28, 50, 56, 94,
        87, 57, 70,  4, 26, 33, 34, 49, 37, 89, 21, 84, 31, 48, 43, 85, 79,
        39, 86, 30, 23, 27, 38, 47, 95, 29, 91, 42, 25, 14, 20, 16, 62,  2,
        45, 15, 12, 90, 76, 52, 32, 40, 24, 98, 66,  7, 63, 83, 99
        ])
    assert_almost_equal(optimizer.results['f0'], 4576.3359328082515, decimal=11)
    assert_almost_equal(optimizer.results['objective'], 927.3480749017356, decimal=11)
    assert optimizer.results['converged']
    assert optimizer.results['niter'] <= solver_options['maxiter']
    assert optimizer.results['niter'] == 14105
    assert optimizer.results['nfev'] == 14106
    assert optimizer.results['nmoves'] == 1240
    assert_almost_equal(optimizer.results['improvement'], 0.797360139527023, decimal=11)

if __name__ == '__main__':
    test_simulated_annealing()
    print("All tests passed!")