import logging

import numpy as np
from lmfit import fit_report
from matplotlib import pyplot as plt

from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from DynamicModels.OdeFitter import OdeFitter
from DynamicModels.OdeModel import OdeModel
import seaborn as sns

from helpers import plot_y_and_y_hat

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)


def test_convert_to_ode(my_ode: OdeModel):
    logging.info(f'{my_ode=}')
    params = np.random.rand(my_ode.nr_params).tolist()
    dydt = my_ode(.65, [0, 1, 0, 0], *params)
    logging.info(f'{dydt=}')
    assert dydt != [0, 1, 0, 0]


def test_fit_ode_to_data(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries):
    n_clusters = 4
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    my_time_series_expressions.keep_only_de_genes(std_cutoff=1.5)
    my_time, my_data = \
        my_time_series_expressions.get_clusters_expressions_with_time(n_clusters)
    initial_sim_fit = OdeFitter(my_ode, my_data, my_time)
    # Note: look into the initial parameter values
    optimal_fit = initial_sim_fit.fit()
    # logging.info(fit_report(optimal_fit))
    print(fit_report(optimal_fit))

    # Next step: simulate the data with these params
    simulated_data = initial_sim_fit.predict_values(optimal_fit.params, my_time)

    plot_y_and_y_hat(y=my_data, y_hat=simulated_data.y, t=simulated_data.t)

    # And try to fit model again
    try_again = OdeFitter(my_ode, simulated_data.y, my_time)
    second_fit = try_again.fit()

    fit_to_simul = try_again.predict_values(second_fit.params, my_time)

    plot_y_and_y_hat(y=simulated_data.y, y_hat=fit_to_simul.y, t=fit_to_simul.t)
    # logging.info(fit_report(second_fit))
    print(fit_report(second_fit))
    # This time the parameters get quite close ...
    assert True
