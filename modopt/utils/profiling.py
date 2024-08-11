from memory_profiler import memory_usage
import time

def time_profiler():
    def decorator(func):
        def wrapper(*args, **kwargs):

            start_time = time.perf_counter()
            results = func(*args, **kwargs)
            exec_time = time.perf_counter() - start_time
            print(f"Execution time: {exec_time} seconds")
            
            return results, exec_time
        return wrapper
    return decorator

# Using memory_profiler to look at memory usage at specific intervals during while the function is running.
# This will consider other memory allocations (e.g.,) that are not directly related to the function.

def profiler(interval=1e-4):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            outputs = {}
            _func = lambda: outputs.update({'results': func(*args, **kwargs)})
            mem_usage = memory_usage(_func, interval=interval)
            
            end_time = time.perf_counter()
            
            mem_extrema  = (min(mem_usage), max(mem_usage))
            time_extrema = (start_time, end_time)
            
            # Calculate memory usage and execution time
            mem_used = mem_extrema[1] - mem_extrema[0]
            exec_time = end_time - start_time
            # print(f"Memory usage: {mem_used} MB")
            # print(f"Execution time: {exec_time} seconds")
            
            return outputs['results'], (mem_used, mem_extrema), (exec_time, time_extrema)
        return wrapper
    return decorator

# # Using guppy to look at only memory allocated by the function relative to the point where setrelheap() is called.
# # The memory used during the function call is ignored and only memory allocated after the function is run is considered.
# # See the 2 examples below for the difference between the two.

# from guppy import hpy

# def profiler(interval=1e-4):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             start_time = time.perf_counter()
            
#             h = hpy()
#             h.setrelheap()
#             results = func(*args, **kwargs)
#             mem_allocations = h.heap()
#             # print(mem_allocations)
#             # print(mem_allocations.size)
    
#             end_time = time.perf_counter()
            
#             # Calculate memory allocated after function is run and execution time
#             mem_used = mem_allocations.size / 1024 / 1024 # MB
#             exec_time = end_time - start_time
#             print(f"Memory allocated: {mem_used} MB")
#             print(f"Execution time: {exec_time} seconds")
            
#             return results, mem_used, exec_time
#         return wrapper
#     return decorator

def my_function():
    a = [i for i in range(1000000)]
    return

@profiler(interval=1e-4) # # No memory is allocated here after function is run
def my_function_decorated1():
    a = [i for i in range(1000000)]
    return 

@profiler(interval=1e-4)  # Memory is allocated here after function is run
def my_function_decorated2():
    a = [i for i in range(1000000)]
    return a

if __name__ == '__main__':
    # Measure memory usage of a function without using the profiler decorator
    mem_usage = memory_usage(my_function)
    print(f"Memory usage: {max(mem_usage) - min(mem_usage)} MB")

    # Measure memory usage of a function using the profiler decorator
    _, mem_used, exec_time = my_function_decorated1()
    print(f"Memory usage: {mem_used} MB")
    print(f"Execution time: {exec_time} seconds")

    _, mem_used, exec_time = my_function_decorated2()
    print(f"Memory usage: {mem_used} MB")
    print(f"Execution time: {exec_time} seconds")