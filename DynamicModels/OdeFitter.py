import logging

import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from DynamicModels.OdeModel import OdeModel


class OdeFitter:
    """Class to estimate parameters for ODEs, given real data"""
    def __init__(self, ode_inference: OdeModel, measured_data: np.ndarray,
                 time_points: np.ndarray):
        self.odes = ode_inference
        self.measured_data = measured_data
        self.time_points = time_points
        self.params = Parameters()
        # TODO how to handle guesses for initial params/constraints?
        # TODO Note that model seems really sensitive to this guess of param_limit.
        #  If you get any error, it could be because this param_limit is too high
        param_limit = .1
        for param_name in ode_inference.get_param_names():
            min_value = -param_limit
            max_value = param_limit
            # Decay rates cannot be negative
            if 'd' in param_name:
                # Min value always at 0 helps speed up solving IVP for some reason?
                min_value = 0.
            # elif 't' in param_name:
            #     min_value = -.2
            #     max_value = 0
            self.params.add(param_name, min=min_value, max=max_value)
        self.init_condition_names = []
        for i, init_value in enumerate(measured_data[:, 0]):
            init_y_name = f'y{i}'
            self.params.add(init_y_name, value=init_value, vary=False)
            self.init_condition_names.append(init_y_name)
        # self.fitting_complete = False

    def loss_function(self, params: Parameters, t: np.ndarray,
                      y_real: np.ndarray, t_start=None, t_end=None) -> np.ndarray:
        """
        Return squared residuals between y_real and a predictions
        at time points t for a given set params.
        """
        if t_start is not None or t_end is not None:
            selected_time_points = (t_start <= t) & (t <= t_end)
            y_real = y_real[:, selected_time_points]
        y_pred = self.predict_values(params, t, t_start, t_end)
        return np.square(y_pred.y - y_real)

    def predict_values(self, params: Parameters, t: np.ndarray, t_start=None,
                       t_end=None) -> OdeResult:
        """Return values at time points t, given a set of params"""
        # TODO make this an OdeInference method, instead of it belonging to this class?
        if t_start is None:
            t_start = min(self.time_points)
        if t_end is None:
            t_end = max(self.time_points)
        if t_start is not None or t_end is not None:
            selected_time_points = (t_start <= t) & (t <= t_end)
            t = t[selected_time_points]
        y0 = [value for name, value in params.valuesdict().items()
              if name in self.init_condition_names]
        tuple_params = [value for name, value in params.valuesdict().items()
                        if name not in self.init_condition_names]
        # Seems to take extremely long sometimes
        y_pred = solve_ivp(self.odes, (t_start, t_end), y0, t_eval=t,
                           args=tuple_params)
        assert y_pred.success, (f"Integration failed: {y_pred.message}"
                                f"\nParams: {params}")
        return y_pred

    def fit(self, t_start=None, t_end=None) -> MinimizerResult:
        output = minimize(self.loss_function,
                          self.params,
                          kws={'t': self.time_points,
                               'y_real': self.measured_data,
                               't_start': t_start,
                               't_end': t_end})
        # logging.info('Fitting complete. Updating params')
        # self.params = output.params
        # self.fitting_complete = True
        return output
        # output.params.pretty_print()

    def set_params(self):
        raise NotImplementedError
