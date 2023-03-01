import logging
from copy import copy

import numpy as np
from lmfit import fit_report, Parameters, Parameter

from DynamicModels.OdeFitter import OdeFitter
from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from helpers import plot_y_and_y_hat


def inference_with_wrong_ode_structure(ode_model: OdeModel):
    # t = np.linspace(0, 24, 20)
    t = np.array([0.25, 0.5, 1., 3., 4., 6., 12., 24.])
    simulation_params = Parameters()
    # Ground truth parameters
    simulation_params.add_many(
        ("beta_1_0", -0.48560429),
        ("delta_0", 0.60382809),
        ("gamma_0", 7.35696320),
        ("heat_temp", 0.40369203),
        ("non_heat_temp", 0.59630797),
        ("delta_1", 1.50103715),
        ("gamma_1", 8.92472880),
        ("beta_0_2", -1.61590940),
        ("beta_1_2", -0.22866582),
        ("beta_3_2", 3.67249005),
        ("delta_2", 1.88822134),
        ("gamma_2", 3.69594181),
        ("beta_1_3", 0.86542422),
        ("beta_2_3", 0.80666603),
        ("delta_3", 1.49367792),
        ("gamma_3", -1.62143388),
        ("y0", 3.19831),
        ("y1", 3.924275),
        ("y2", 2.327565),
        ("y3", 3.115082)
    )
    ode_model_with_extra_link = copy(ode_model)
    new_param_name = ode_model_with_extra_link.add_regulator_to_module(
        0, origin_module_idx=2)
    simulation_params.add(name=new_param_name, value=-.02)
    init_condition_names = ['y0', 'y1', 'y2', 'y3']
    simulated_data = ode_model_with_extra_link.calculate_solution(
        simulation_params, t, init_condition_names)
    plot_y_and_y_hat(y_real=simulated_data.y, t_real=t)

    # Next step: simulate the data with these params
    simulation_fitter = OdeFitter(ode_model, simulated_data.y, t)
    slightly_different_params = simulation_params.copy()
    slightly_different_params['beta_2_3'] = Parameter(name='beta_2_3', value=1)
    # slightly_different_params['heat_temp'] = Parameter('heat_temp', expr='1 - non_heat_temp')
    simulation_fitter.params = slightly_different_params
    fit_outcome = simulation_fitter.fit()
    logging.info(fit_report(fit_outcome))
    optimal_fit = simulation_fitter.show_current_best_fit(t)
    plot_y_and_y_hat(y_real=simulated_data.y, t_real=t,
                     model_fit=optimal_fit)
    #
    # # And try to fit model again
    # try_again = OdeFitter(my_ode, simulated_data.y, simulated_data.t)
    # second_fit = try_again.fit()
    # fit_to_simul = try_again.predict_values(second_fit.params, simulated_data.t)
    #
    # plot_y_and_y_hat(y_real=simulated_data.y, t_real=simulated_data.t,
    #                  model_fit=fit_to_simul)
    # logging.info(fit_report(second_fit))
    # # print(fit_report(second_fit))
