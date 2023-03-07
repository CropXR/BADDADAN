import logging
from typing import Literal

import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from DynamicModels.OdeModel import OdeModel


class OdeFitter:
    """Class to estimate parameters for ODEs, given real data"""

    def __init__(self, ode_model: OdeModel, measured_data: np.ndarray,
                 time_points: np.ndarray):
        self.odes = ode_model
        self.measured_data = measured_data
        self.has_been_fitted = False
        self.time_points = time_points
        self.params = Parameters()
        self.param_limit = 10
        for param_name in ode_model.get_param_names():
            min_value = -self.param_limit
            max_value = self.param_limit
            if 'delta' in param_name:
                # Decay rates cannot be negative
                min_value = 0.
            self.params.add(param_name,
                            value=np.random.uniform(min_value / 3,
                                                    max_value / 3),
                            min=min_value, max=max_value)
        self.init_condition_names = []
        for i, init_value in enumerate(measured_data[:, 0]):
            init_y_name = f'y{i}'
            self.params.add(init_y_name, value=init_value, vary=False)
            self.init_condition_names.append(init_y_name)

        # Restrain non_heat_temp and heat_temp so they always sum to one
        self.params.add('non_heat_temp', value=np.random.rand(), min=0, max=1)
        self.params.add('heat_temp', expr='1 - non_heat_temp')
        # # Does doing this line below stop the equation from exploding?
        # self.params.add('heat_temp', value=np.random.rand(), min=0, max=1)

    def loss_function(self, params: Parameters, t: np.ndarray,
                      y_real: np.ndarray, t_start=None,
                      t_end=None, return_scalar=False) -> np.ndarray | float:
        """Return squared residuals between y_real and predictions
        at time points t for a given set params.
        """
        # Get names of 'parameters' that are the initial conditions
        init_condition_names = self.init_condition_names
        y_pred = self.odes.calculate_solution(params, t, init_condition_names,
                                              t_start, t_end)
        if return_scalar:
            return float(np.mean(np.square(y_pred.y - y_real)))
        loss = np.square(y_pred.y - y_real)
        return loss

    def calculate_current_best_fit(self, t: np.ndarray) -> OdeResult:
        """Calculate the solution of the ODEs for the set of
         parameters that fitting found
         """
        assert self.has_been_fitted, "First fit ODE to the data using .fit()"
        return self.odes.calculate_solution(self.params, t,
                                            self.init_condition_names)

    def fit(self, t_start: float = None,
            t_end: float = None,
            method: Literal['lbfgs', 'bfgs',
                            'differential_evolution',
                            'basinhopping', 'shgo'] = 'lbfgs') -> MinimizerResult:
        """Find optimal parameters for ODE

        :param t_start: Timepoint to start simulation, defaults to lowest time point
        :param t_end:  Timepoint to end simulation, defaults to highest time point
        :param method: Optimisation method to use
        :return: result of minimisation
        """
        match method:
            case 'lbfgs' | 'bfgs':
                result = minimize(self.loss_function,
                                  self.params,
                                  method=method,
                                  kws={'t': self.time_points,
                                       'y_real': self.measured_data,
                                       't_start': t_start,
                                       't_end': t_end,
                                       'return_scalar': True},
                                  options=dict(disp=1),
                                  )
            case 'differential_evolution':
                result = minimize(self.loss_function,
                                  self.params,
                                  method='differential_evolution',
                                  kws={'t': self.time_points,
                                       'y_real': self.measured_data,
                                       't_start': t_start,
                                       't_end': t_end,
                                       'return_scalar': True},
                                  max_nfev=20_000,
                                  workers=1,
                                  # popsize=4,
                                  # polish=False,
                                  disp=True
                                  )
            case 'basinhopping':
                result = minimize(self.loss_function,
                                  self.params,
                                  method='basinhopping',
                                  kws={'t': self.time_points,
                                       'y_real': self.measured_data,
                                       't_start': t_start,
                                       't_end': t_end},
                                  disp=True,
                                  niter=300
                                  )
            case 'shgo':
                result = minimize(self.loss_function,
                                  self.params,
                                  method='shgo',
                                  kws={'t': self.time_points,
                                       'y_real': self.measured_data,
                                       't_start': t_start,
                                       't_end': t_end,
                                       'return_scalar': True},
                                  max_nfev=10_000,
                                  options=dict(disp=True)
                                  )
            case _:
                raise NotImplementedError(f'Optimisation method: {method} '
                                          f'is currently not supported')

        self.params = result.params
        self.has_been_fitted = True
        return result

    def thickening_thinning(self, nr_rounds: int) -> MinimizerResult:
        logging.warning(
            'THICKENING THINNING IS NOT PROPERLY IMPLEMENTED AT THE MOMENT')
        for i in range(nr_rounds):
            minimizer_result = self.fit()
            best_params = minimizer_result.params
            # Get which module performs worst
            worst_module = self.get_worst_module(best_params)
            self.do_thickening(worst_module)
            # TODO implement thinning
            logging.info(f'Iteration {i}\nLoss: {minimizer_result.residual}\n')
        return self.fit()

    def do_thickening(self, module_idx: int):
        new_param_name = self.odes.add_regulator_to_module(module_idx)
        if new_param_name:
            min_value = -self.param_limit
            max_value = self.param_limit
            self.params.add(new_param_name,
                            value=np.random.uniform(min_value / 3,
                                                    max_value / 3),
                            min=min_value, max=max_value)

    def get_worst_module(self, best_params: Parameters) -> int:
        loss_per_module = np.sum(
            self.loss_function(best_params, self.time_points,
                               self.measured_data), axis=1)
        worst_module_idx = np.argmax(loss_per_module)
        return worst_module_idx

    def get_best_module(self, best_params: Parameters) -> int:
        loss_per_module = np.sum(
            self.loss_function(best_params, self.time_points,
                               self.measured_data), axis=1)
        best_module_idx = np.argmin(loss_per_module)
        return best_module_idx

    def do_thinning(self):
        raise NotImplementedError
