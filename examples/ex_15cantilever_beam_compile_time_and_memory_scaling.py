'''Scaling study on optimizers and AMLS (Cantilever)'''

import matplotlib.pyplot as plt

from modopt.utils.profiling import profiler

from examples.ex_15_1cantilever_beam_fd import get_problem as get_fd_prob
from examples.ex_15_2cantilever_beam_casadi import get_problem as get_ca_prob
from examples.ex_15_3cantilever_beam_csdl import get_problem as get_csdl_prob
from examples.ex_15_4cantilever_beam_jax import get_problem as get_jax_prob
from examples.ex_15_5cantilever_beam_openmdao import get_problem as get_om_prob

num_elements = [20, 50, 100, 200, 300, 500]


get_probs = [get_fd_prob, get_ca_prob, get_csdl_prob, get_jax_prob, get_om_prob]
methods   = ['FD', 'CasADi', 'CSDL', 'Jax', 'OpenMDAO']

if __name__ == '__main__':
    data = {}
    for i, n_el in enumerate(num_elements):
        for get_prob, method in zip(get_probs, methods):
            interval = 1e0

            _compile_prob = profiler(interval=interval)(lambda n: get_prob(n))
            prob, compile_mem, compile_time = _compile_prob(n_el)

            print('='*50)
            print('Method:', n_el, method)
            print('-'*50)

            print('compile_memory', compile_mem)
            print('compile_time', compile_time)

            data[n_el, method] = {'memory': compile_mem, 'time': compile_time}

            if method in ['CasADi', 'Jax']:
                print('='*50)
                print('method:', n_el, method+'-2')
                print('-'*50)

                _compile_prob = profiler(interval=interval)(lambda n: get_prob(n, order=2))
                prob, compile_mem, compile_time = _compile_prob(n_el)
                
                print('compile_memory', compile_mem)
                print('compile_time', compile_time)

                data[n_el, method+'-2'] = {'memory': compile_mem, 'time': compile_time}

    print('data:', data)
    
    import pickle
    with open('cantilever_compile_mem_and_times.pkl', 'wb') as file:
        pickle.dump(data, file)
    with open('cantilever_compile_mem_and_times.pkl', 'rb') as file:
        loaded_data = pickle.load(file)

    print('loaded_data:', loaded_data)

# Plot memory scaling - AMLs (for IPOPT/-2)
plt.figure()
for aml in methods:
    y_data = [data[n_el, aml]['memory'] for n_el in num_elements]
    plt.semilogy(num_elements, y_data, label=f"{aml}")

for aml in ['CasADi', 'Jax']:
    x_data = num_elements
    y_data = [data[n_el, aml+'-2']['memory'] for n_el in num_elements]
    plt.semilogy(num_elements, y_data, label=f"{aml}-2")

plt.xlabel('Number of elements')
plt.ylabel('Memory usage [MB]')
plt.title('Compile memory scaling of AMLs')
plt.legend()
plt.grid()
plt.savefig(f'cantilever_aml_compile_memory_scaling_semilogy.pdf')
plt.close()


# Plot time scaling - AMLs (for IPOPT/-2)
plt.figure()
for aml in methods:
    y_data = [data[n_el, aml]['time'] for n_el in num_elements]
    plt.semilogy(num_elements, y_data, label=f"{aml}")

for aml in ['CasADi', 'Jax']:
    x_data = num_elements
    y_data = [data[n_el, aml+'-2']['time'] for n_el in num_elements]
    plt.semilogy(num_elements, y_data, label=f"{aml}-2")

plt.xlabel('Number of elements')
plt.ylabel('Compile time [MB]')
plt.title('Compile time scaling of AMLs')
plt.legend()
plt.grid()
plt.savefig(f'cantilever_aml_compile_time_scaling_semilogy.pdf')
plt.close()