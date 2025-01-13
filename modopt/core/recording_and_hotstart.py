# Decorators for recording, hot-starting and visualizing the optimization callbacks.

import numpy as np
import warnings
import functools

def hot_start(in_names, out_names):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._hot_start_mode:
                if self._callback_count < self._num_callbacks_found:
                    group_name      = f'callback_{self._callback_count}'
                    group           = self._hot_start_record[group_name]
                    group_in_names  = list(group['inputs'].keys())
                    group_out_names = list(group['outputs'].keys())
                    rtol            = self._hot_start_tol[0]
                    atol            = self._hot_start_tol[1]
                    for in_name, arg in zip(in_names, args):
                        if in_name not in group_in_names:
                            warnings.warn(f"Input {in_name} not found in the record '{group_name}'. "\
                                          f"Running normal function evaluation ...")
                            return func(self, *args, **kwargs)
                        if not np.allclose(group['inputs'][in_name], arg, rtol=rtol, atol=atol):
                            warnings.warn(f"Input {in_name} does not match the recorded input within the tolerance for the record {group_name}. " \
                                          f"Running normal function evaluation ...")
                            return func(self, *args, **kwargs)
                    for out_name in out_names:
                        if out_name not in group_out_names:
                            warnings.warn(f"Output {out_name} not found in the record '{group_name}'. " \
                                          f"Running normal function evaluation ...")
                            return func(self, *args, **kwargs)
                    # print(f"Reusing callback {self._callback_count}: {in_names} -> {out_names}")
                    # If all inputs and outputs are found in the record, return the recorded outputs
                    outputs = tuple(group['outputs'][out_name][()] for out_name in out_names)
                    self._reused_callback_count += 1
                    return outputs if len(out_names) > 1 else outputs[0]
                else:
                    self._hot_start_mode = False # Turn off hot start mode
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)
        
        # # Manually copying metadata when not using functools.wraps
        # # This is important as Sphinx needs to get the correct metadata for autodoc
        # wrapper.__name__ = func.__name__
        # wrapper.__doc__ = func.__doc__
        # wrapper.__module__ = func.__module__
        return wrapper
    return decorator

def record(in_names, out_names):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            results = func(self, *args, **kwargs)
            if self._record:
                # print(f"Recording callback {self._callback_count}: {in_names} -> {out_names}")
                group   = self._record.create_group(f'callback_{self._callback_count}')
                inputs  = group.create_group('inputs')
                outputs = group.create_group('outputs')
                for in_name, arg in zip(in_names, args):
                    inputs[in_name] = arg
                
                if len(out_names) > 1:
                    for out_name, result in zip(out_names, results):
                        # print(f"Recording output {out_name}: {result}")
                        # For compute_all used for unconstrained optimization
                        if result is None:
                            outputs[out_name] = 'None'
                        else:
                            outputs[out_name] = result
                else:
                    # print(f"Recording output {out_names[0]}: {results}")
                    outputs[out_names[0]] = results

            # Update the visualizer [TODO: can be moved out of recording decorator]
            if self._visualizer:
                in_var_dict  = dict(zip(in_names, args))
                out_var_dict = dict(zip(out_names, results)) if len(out_names) > 1 else {out_names[0]: results}
                var_dict     = {**in_var_dict, **out_var_dict}
                callback_var_dict = {f'callback_{var}': var_dict[var] for var in var_dict}
                self._visualizer.update_plot(callback_var_dict)

            self._callback_count += 1
            self._obj_count  += 'obj' in out_names
            self._grad_count += 'grad' in out_names
            self._con_count  += 'con' in out_names
            self._jac_count  += 'jac' in out_names
            self._hess_count += ('obj_hess' in out_names or 'lag_hess' in out_names)
            return results
        
        # # Manually copying metadata when not using functools.wraps
        # # This is important as Sphinx needs to get the correct metadata for autodoc
        # wrapper.__name__ = func.__name__
        # wrapper.__doc__ = func.__doc__
        # wrapper.__module__ = func.__module__
        return wrapper
    return decorator