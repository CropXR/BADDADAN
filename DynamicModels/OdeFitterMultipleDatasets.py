import copy
import logging
from pathlib import Path
from typing import List, Dict, Literal

from lmfit import Parameters, minimize, fit_report
from lmfit.minimizer import MinimizerResult
from matplotlib import pyplot as plt

from DynamicModels.OdeFitter import OdeFitter
from DynamicModels.OdeLocalParameters import OdeLocalParameters
from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from helpers import check_all_identical_lists, plot_y_and_y_hat


class OdeFitterMultipleDatasets:
    """Class for fitting to multiple datasets"""
    def __init__(self, ode_model: OdeModel,
                 datasets: List[ExpressionMatrixTimeSeries],
                 custom_params_per_dataset: Dict[ExpressionMatrixTimeSeries,
                                                 OdeLocalParameters],
                 param_limit: float = 800.,
                 method: Literal['lbfgs', 'bfgs',
                                 'differential_evolution',
                                 'basinhopping', 'shgo'] = 'lbfgs'):
        """
        :param ode_model: The ODE model used for fitting.
        :param datasets: A list of ExpressionMatrixTimeSeries objects, one for
         each dataset.
        :param custom_params_per_dataset: A dictionary mapping each dataset
         to its custom parameters as OdeLocalParameters object.
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
                0, aggregation_method='pca')
            u_t_for_dataset = custom_params_per_dataset[dataset].u_t
            fitter = OdeFitter(copy.deepcopy(ode_model), data, time, param_limit=param_limit,
                               u_t_function=u_t_for_dataset,
                               method=method)
            self.all_fitters.append(fitter)
        logging.info(f'Created {len(self.all_fitters)} fitters '
                     f'to be fitted simultaneously.')

        # Master params are the parameters that are the same between all fitters
        self._master_params = self._get_master_params_from_fitters()

        # Custom (or local) parameters are the parameters that are different between the fitters.
        # First off; the initial values are different between fits
        self.local_param_names = ode_model.get_init_condition_names()
        # The custom parameters shouldn't be in the master parameters
        # for param_name in ode_model.get_init_condition_names():
        #     self._master_params.pop(param_name)

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
        # all y0 values should be variable too of course ya knobhead
        for fitter in self.all_fitters:
            loss = fitter.loss_function(
                params, fitter.time_points,
                fitter.measured_data, custom_param_names, return_scalar=True)
            norm_loss = loss / len(fitter.time_points)
            all_loss.append(norm_loss)
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
                                  self._master_params,
                                  method=self.method,
                                  kws=dict(custom_param_names=self.local_param_names),
                                  options=dict(disp=1, maxiter=max_iter,
                                               maxfun=1e99,
                                               )
                                  )
            case _:
                raise NotImplementedError(f'Optimisation method: {self.method} '
                                          f'is currently not supported')

        # For some reason the local parameters show up again. So remove
        # them from the final fitting parameters again
        for name in self.local_param_names:
            result.params.pop(name)
        self.master_params = result.params
        self.has_been_fitted = True
        logging.info('Global parameters: ')
        logging.info(fit_report(result))

        for fitter in self.all_fitters:
            logging.info(f'Local parameters of {fitter}')
            logging.info([fitter.params[p] for p in self.local_param_names])

        return result

    def calculate_current_best_fits(self, out_path: Path = None):
        """Calculate the solution of the ODEs for all conditions to which
        they were fitted
         """
        fig, axs = plt.subplots(len(self.all_fitters), 2, sharey='all')
        logging.debug([(i, j)
                       for i, j in zip(self.all_fitters[0].params.values(),
                                       self.all_fitters[1].params.values())
                       ]
                      )
        for i, fitter in enumerate(self.all_fitters):
            i *= 2
            ax = axs.flatten()[i:i+2]
            pred = fitter.calculate_current_best_fit(fitter.time_points)
            real = fitter.measured_data
            plot_y_and_y_hat(real, fitter.time_points, pred, axs=ax)
        plt.tight_layout()
        # plt.savefig(out_path)
        plt.show()

    @property
    def master_params(self) -> Parameters:
        return self._master_params

    @master_params.setter
    def master_params(self, new_parameters: Parameters):
        """Set the master parameters (the ones shared between all fitters)

        Can be done if you have prior knowledge, or if you have finished
        fitting the ODEs.

        Here, a setter method is used because you need to update all the
        parameters of the underlying fitters as well.
        """
        # Only update the values, not the upper/lower limit
        for param_name in new_parameters.valuesdict():
            if param_name in self._master_params:
                self._master_params[param_name].set(
                    value=new_parameters[param_name].value)
            else:
                logging.warning(f'{param_name} is not a global parameters, so cannot be set here')
                continue
        for fitter in self.all_fitters:
            fitter.params.update(self._master_params)
            if 'non_heat_temp' in new_parameters:
                logging.info(
                    'Setting non_heat_temp as local global parameter')
                fitter.params['non_heat_temp'].set(value=new_parameters['non_heat_temp'].value)
            fitter.has_been_fitted = True

    def _get_master_params_from_fitters(self) -> Parameters:
        """
        If ranges are slightly different between parameters,
        ensure that the global parameters have the largest range.
        """
        master_params = self.all_fitters[0].params.copy()
        for fitter in self.all_fitters:
            for param_name, param_object in fitter.params.items():
                master_params[param_name].min = min(param_object.min,
                                                    master_params[param_name].min)
                master_params[param_name].max = max(param_object.max,
                                                    master_params[param_name].max)
        return master_params
