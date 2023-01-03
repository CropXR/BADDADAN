import numpy as np
from lmfit import Parameters, minimize
from scipy.integrate import solve_ivp

from OdeInference import OdeInference


class OdeFitter:
    """Class to estimate parameters for ODEs, given real data"""
    def __init__(self, ode_inference: OdeInference, measured_data: np.ndarray, time_points: np.ndarray):
        self.odes = ode_inference
        self.measured_data = measured_data
        self.time_points = time_points
        self.params = Parameters()
        # TODO how to handle guesses for initial params/constraints?
        for param_name in ode_inference.get_param_names():
            self.params.add(param_name, min=-1e90, max=1e90)
        self.init_condition_names = []
        for i, init_value in enumerate(measured_data[:, 0]):
            init_y_name = f'y{i}'
            self.params.add(init_y_name, value=init_value, vary=False)
            self.init_condition_names.append(init_y_name)

    def set_param_limits(self):
        raise NotImplementedError

    def loss_function(self, params: Parameters, t, y_real, my_func, t_start=None, t_end=None):
        # TODO THIS IS BROKENNNNN / hardcoded / too hacky for now. Also add type hints
        """Return squared residuals between y_real and a prediction of my_func
        at time points t for a given set params.
        """
        if t_start is None:
            t_start = min(self.time_points)
        if t_end is None:
            t_end = max(self.time_points)
        if t_start is not None or t_end is not None:
            selected_time_points = (t_start <= t) & (t <= t_end)
            t = t[selected_time_points]
            y_real = y_real[:, selected_time_points]
        y0 = [value for name, value in params.valuesdict().items()
              if name in self.init_condition_names]
        tuple_params = [value for name, value in params.valuesdict().items()
                        if name not in self.init_condition_names]
        print('Solving IVP')
        # Seems to take extremely long when values are infinitely large
        y_pred = solve_ivp(my_func, (t_start, t_end), y0, t_eval=t,
                           args=tuple_params, method='RK45')
        assert y_pred.status >= 0, f"Integration failed {y_pred.message}"
        return np.square(y_pred.y - y_real)

    def fit(self, t_start=None, t_end=None):
        # TODO pass arguments as kwargs instead
        output = minimize(self.loss_function,
                          self.params,
                          method='least_squares',
                          kws={'t': self.time_points,
                               'y_real': self.measured_data,
                               'my_func': self.odes,
                               't_start': t_start,
                               't_end': t_end},
                          verbose=2)
        output.params.pretty_print()
