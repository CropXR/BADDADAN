import logging

from typing import Literal, List, Dict, Generator

import numpy as np
import pandas as pd
from lmfit import Parameters, minimize, fit_report
from lmfit.minimizer import MinimizerResult
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import seaborn as sns

from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from helpers import check_all_identical_lists, plot_y_and_y_hat


class OdeFitter:
    """Class to estimate parameters for ODEs, given real data"""

    def __init__(self, ode_model: OdeModel, measured_data: np.ndarray,
                 time_points: np.ndarray, heat_end_time: float,
                 param_limit: float = 800.,
                 method: Literal['lbfgs', 'bfgs',
                                 'differential_evolution',
                                 'basinhopping', 'shgo'] = 'lbfgs'):
        self.odes = ode_model
        self.measured_data = measured_data
        self.has_been_fitted = False
        self.method = method
        self.time_points = time_points
        self.param_limit = param_limit

        # Set parameters
        self.params = Parameters()
        for param_name in self.odes.get_param_names():
            min_value = -self.param_limit
            max_value = self.param_limit
            if 'delta' in param_name or 'k_' in param_name:
                # Decay rates cannot be negative
                min_value = 0.
            if self.odes.is_nonlinear and 'beta_' in param_name:
                # Beta values cannot be negative in nonlinear model
                min_value = 0.
            if 'gamma' in param_name:
                min_value = -10
                max_value = 10

            self.params.add(param_name,
                            value=np.random.uniform(min_value,
                                                    max_value),
                            min=min_value, max=max_value)
        self.init_condition_names = []
        for i, init_value in enumerate(self.measured_data[:, 0]):
            init_y_name = f'y{i}'
            self.params.add(init_y_name, value=init_value, vary=True, min=0,
                            max=max(self.measured_data[:, 0]) * 2)
            self.init_condition_names.append(init_y_name)
        # Add moment when heatstress ends
        self.params.add('heat_end_time', value=heat_end_time, vary=False)
        # Restrain non_heat_temp and heat_temp so they always sum to one
        self.params.add('non_heat_temp', value=np.random.rand(), min=0.1, max=.9)
        self.params.add('heat_temp', expr='1 - non_heat_temp')

    def loss_function(self, params: Parameters, t: np.ndarray,
                      y_real: np.ndarray,
                      custom_param_names: set[str] = None, return_scalar=False
                      ) -> np.ndarray | float:
        """Return squared residuals between y_real and predictions
        at time points t for a given set of params.

        :param params: The parameters used for prediction.
        :param t: The time points at which predictions are made.
        :param y_real: The actual values at the time points.
        :param custom_param_names: Names of custom parameters to include in
         the calculations. (optional)
        :param return_scalar: Whether to return the mean squared loss
         as a scalar. (default: False)

        :return: The squared residuals between y_real and predictions, or
        the mean squared loss if return_scalar is True.
        """
        # Get names of 'parameters' that are the initial conditions
        init_condition_names = self.init_condition_names
        if custom_param_names:
            # Parameters that should not be taken from 'params' but should use
            # the value that is in the local 'self.params'.
            for custom_param in custom_param_names:
                params.add(self.params[custom_param])
        y_pred = self.odes.calculate_solution(params, t, init_condition_names)
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

    def fit(self, max_iter=None) -> MinimizerResult:
        """Find optimal parameters for ODE

        :param max_iter: For lbfgs/bfgs, set the maximum number of iterations
        :return: result of minimisation
        """
        if max_iter is None:
            max_iter = 1000
        else:
            assert self.method in ['lbfgs', 'bfgs'], \
                'Can only enter maxiter for BFGS/LBFGS methods currently.'

        match self.method:
            case 'lbfgs' | 'bfgs':
                result = minimize(self.loss_function,
                                  self.params,
                                  method=self.method,
                                  kws={'t': self.time_points,
                                       'y_real': self.measured_data,
                                       'return_scalar': True},
                                  options=dict(disp=1, maxiter=max_iter,
                                               maxfun=1e99),
                                  )
            case 'differential_evolution':
                result = minimize(self.loss_function,
                                  self.params,
                                  method='differential_evolution',
                                  kws={'t': self.time_points,
                                       'y_real': self.measured_data,
                                       'return_scalar': True},
                                  # max_nfev=20_000,
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
                                       'y_real': self.measured_data},
                                  disp=True,
                                  niter=300
                                  )
            case 'shgo':
                result = minimize(self.loss_function,
                                  self.params,
                                  method='shgo',
                                  kws={'t': self.time_points,
                                       'y_real': self.measured_data,
                                       'return_scalar': True},
                                  max_nfev=10_000,
                                  options=dict(disp=True)
                                  )
            case _:
                raise NotImplementedError(f'Optimisation method: {self.method} '
                                          f'is currently not supported')

        self.params = result.params
        self.has_been_fitted = True
        logging.info(fit_report(result))
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

    def do_thinning(self):
        raise NotImplementedError

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

    def plot_hill_equation_range(self, t: np.ndarray) -> None:
        """For current fit, see outcomes of the hill equations.

        Basically, check if they are not just one value, but
        exhibit dynamic behaviour over time.

        :param t: Time points at which hill equations should be evaluated
        :return: Plots of raw and normalised outcomes of the hill
         equations in the fit
        """
        current_fit = self.calculate_current_best_fit(t)
        # Iterate over all formulas
        out_list = []
        for formula in self.odes.formula_per_module:
            if not formula.regulator_names:
                # Module is not regulated by any other modules
                continue
            for outcome in formula.get_hill_equation_outcomes(current_fit,
                                                              self.params, t):
                out_list.append(outcome)

        out_df = pd.DataFrame.from_records(out_list,
                                           columns=['module_w_regulator',
                                                    't', 'outcome', 'norm_outcome'])
        sns.relplot(data=out_df, y='outcome', x='t', row='module_w_regulator',
                    kind='line', facet_kws=dict(sharey=False))
        plt.show()
        sns.relplot(data=out_df, y='norm_outcome', x='t', row='module_w_regulator',
                    kind='line', facet_kws=dict(sharey=True))
        plt.show()


class OdeFitterMultipleDatasets:
    """Class for fitting to multiple datasets"""
    def __init__(self, ode_model: OdeModel,
                 datasets: List[ExpressionMatrixTimeSeries],
                 custom_params_per_dataset: Dict[ExpressionMatrixTimeSeries,
                                                 Parameters],
                 param_limit: float = 800.,
                 method: Literal['lbfgs', 'bfgs',
                                 'differential_evolution',
                                 'basinhopping', 'shgo'] = 'lbfgs'):
        """
        :param ode_model: The ODE model used for fitting.
        :param datasets: A list of ExpressionMatrixTimeSeries objects, one for
         each datasets.
        :param custom_params_per_dataset: A dictionary mapping each dataset
         to its custom parameters.
        :param param_limit: The parameter limit. Default is 800.
        :param method: The optimization method to be used. Default is 'lbfgs'.
        """

        self.method = method
        self.has_been_fitted = False

        # Combine all OdeFitter objects
        self.all_fitters = []
        for dataset in datasets:
            assert dataset.has_been_clustered
            time, data = dataset.get_clusters_expressions_with_time(
                0, aggregation_method='mean')
            # TODO temporary fix because I'm lazy, might have to be changed later
            heat_end = custom_params_per_dataset[dataset]['heat_end_time']
            fitter = OdeFitter(ode_model, data, time,
                               heat_end_time=heat_end,
                               param_limit=param_limit,
                               method=method)
            self.all_fitters.append(fitter)
        logging.info(f'Created {len(self.all_fitters)} fitters '
                     f'to be fitted simultaneously.')

        # Master params are the parameters that are the same between all fitters
        self.master_params = self.all_fitters[0].params.copy()

        # Custom parameters are the parameters that are different between the fitters.
        self.custom_param_names = self.get_local_parameter_names(
            custom_params_per_dataset)
        # The custom parameters shouldn't be in the master parameters
        for param_name in self.custom_param_names:
            self.master_params.pop(param_name)

        # TODO how to handle the non_heat temp? Atm assumes the same between
        #  two samples.

    @staticmethod
    def get_local_parameter_names(
            custom_params_per_dataset: Dict[ExpressionMatrixTimeSeries, Parameters]
    ) -> set:
        """Get all custom parameter names from the given custom_params_per_dataset.

        :param custom_params_per_dataset: A dictionary mapping each dataset
         to its custom parameters.
        :return: Set which contains the names of the custom (=local) params.
        """
        output = []
        for custom_parameters in custom_params_per_dataset.values():
            param_names_one_dict = [k for k in custom_parameters.keys()]
            output.append(param_names_one_dict)
        # Assert that all dictionaries contain the same custom parameter names
        assert check_all_identical_lists(output)
        return set(output[0])

    def loss_on_multiple_datasets(self, params: Parameters, custom_param_names: set[str] = None) -> float:
        """Fit multiple time series and return the total loss over all datasets.

        :param params: Parameters for which loss should be calculated.
        :param custom_param_names: Names of additional local parameters
        to include in the calculations. (optional)

        :return: Total loss as a float.
        """
        all_loss = []
        # all y0 values should be variable, too of course ya knobhead
        for fitter in self.all_fitters:
            loss = fitter.loss_function(
                params, fitter.time_points,
                fitter.measured_data, custom_param_names, return_scalar=True)
            all_loss.append(loss)
        return sum(all_loss)

    def fit(self, max_iter=None) -> MinimizerResult:
        """Find optimal parameters for ODE

        :param max_iter: For lbfgs/bfgs, set the maximum number of iterations
        :return: Result of the minimization as a MinimizerResult object.
        """
        if max_iter is None:
            max_iter = 1000
        else:
            assert self.method in ['lbfgs', 'bfgs'], \
                'Can only enter maxiter for BFGS/LBFGS methods currently.'

        match self.method:
            case 'lbfgs' | 'bfgs':
                result = minimize(self.loss_on_multiple_datasets,
                                  self.master_params,
                                  method=self.method,
                                  kws=dict(custom_param_names=self.custom_param_names),
                                  options=dict(disp=1, maxiter=max_iter,
                                               maxfun=1e99,
                                               )
                                  )
            case _:
                raise NotImplementedError(f'Optimisation method: {self.method} '
                                          f'is currently not supported')

        # For some reason the custom parameters show up again. So remove
        # them from the final fitting parameters again
        for name in self.custom_param_names:
            result.params.pop(name)
        self.master_params = result.params
        self.has_been_fitted = True
        for fitter in self.all_fitters:
            fitter.has_been_fitted = True
        logging.info(fit_report(result))
        return result

    def calculate_current_best_fits(self):
        """Calculate the solution of the ODEs for all conditions to which
        they were fitted
         """
        fig, axs = plt.subplots(len(self.all_fitters), 2, sharey='all')
        for i, fitter in enumerate(self.all_fitters):
            i *= 2
            ax = axs.flatten()[i:i+2]
            pred = fitter.calculate_current_best_fit(fitter.time_points)
            real = fitter.measured_data
            plot_y_and_y_hat(real, fitter.time_points, pred, axs=ax)
        plt.tight_layout()
        plt.show()

