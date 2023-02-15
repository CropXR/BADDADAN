import logging
from typing import Literal

import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from DynamicModels.OdeModel import OdeModel
from helpers import de_print_fun


class OdeFitter:
    """Class to estimate parameters for ODEs, given real data"""

    def __init__(self, ode_model: OdeModel, measured_data: np.ndarray,
                 time_points: np.ndarray):
        self.odes = ode_model
        self.measured_data = measured_data
        self.time_points = time_points
        self.params = Parameters()
        # TODO how to handle guesses for initial params/constraints?
        param_limit = 10
        for param_name in ode_model.get_param_names():
            min_value = -param_limit
            max_value = param_limit
            # Decay rates cannot be negative
            if 'delta' in param_name:
                # Min value always at 0 helps speed up solving IVP for some reason?
                min_value = 0.
            # elif 't' in param_name:
            #     min_value = -.2
            #     max_value = 0
            self.params.add(param_name, value=np.random.uniform(-1, 1),
                            min=min_value, max=max_value)
        self.init_condition_names = []
        for i, init_value in enumerate(measured_data[:, 0]):
            init_y_name = f'y{i}'
            self.params.add(init_y_name, value=init_value, vary=False)
            self.init_condition_names.append(init_y_name)

        # Restrain non_heat_temp and heat_temp so they always sum to one
        self.params.add('non_heat_temp', value=np.random.rand(), min=0, max=1)
        self.params.add('heat_temp', expr='1 - non_heat_temp')
        # self.fitting_complete = False

    def loss_function(self, params: Parameters, t: np.ndarray,
                      y_real: np.ndarray, t_start=None,
                      t_end=None) -> np.ndarray:
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
        param_dict = {
            name: value for name, value in params.valuesdict().items()
            if name not in self.init_condition_names
        }
        y_pred = solve_ivp(self.odes, (t_start, t_end), y0, t_eval=t,
                           args=[param_dict])
        assert y_pred.success, (f"Integration failed: {y_pred.message}"
                                f"\nParams: {params}")
        return y_pred

    def fit(self, t_start: float = None, t_end: float = None,
            method: Literal['lbfgs', 'differential_evolution',
                            'basinhopping'] = 'lbfgs') -> MinimizerResult:
        """Find optimal parameters for ODE

        :param t_start: Timepoint to start simulation, defaults to lowest time point
        :param t_end:  Timepoint to end simulation, defaults to highest time point
        :param method: Optimisation method to use
        :return: result of minimisation
        """
        match method:
            case 'lbfgs':
                return minimize(self.loss_function,
                                self.params,
                                method='lbfgs',
                                kws={'t': self.time_points,
                                     'y_real': self.measured_data,
                                     't_start': t_start,
                                     't_end': t_end}
                                )
            case 'differential_evolution':
                return minimize(self.loss_function,
                                self.params,
                                method='differential_evolution',
                                kws={'t': self.time_points,
                                     'y_real': self.measured_data,
                                     't_start': t_start,
                                     't_end': t_end},
                                fit_kws={"callback": de_print_fun,
                                         "polish": False,
                                         "popsize": 4,
                                         "workers": 4,
                                         "maxiter": 2}
                                )
            case 'basinhopping':
                return minimize(self.loss_function,
                                self.params,
                                method='basinhopping',
                                kws={'t': self.time_points,
                                     'y_real': self.measured_data,
                                     't_start': t_start,
                                     't_end': t_end},
                                # fit_kws={'disp': True,
                                #          'niter': 5}
                                )
            case _:
                raise NotImplementedError(f'Optimisation method: {method} '
                                          f'is currently not supported')

    def set_params(self):
        raise NotImplementedError
