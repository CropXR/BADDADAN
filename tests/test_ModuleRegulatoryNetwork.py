import logging

import numpy as np
from lmfit import fit_report
from matplotlib import pyplot as plt

from ExpressionMatrix import ExpressionMatrixTimeSeries
from dynamic_models.OdeFitter import OdeFitter
from dynamic_models.OdeModel import OdeModel
import seaborn as sns

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)



def test_convert_to_ode(my_ode: OdeModel):
    logging.info(f'{my_ode=}')
    params = np.random.rand(my_ode.nr_params).tolist()
    dydt = my_ode(None, [0, 1, 0, 0], *params)
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
    for i, row in enumerate(simulated_data.y):
        plt.plot(simulated_data.t, row, label=str(i))
    plt.legend()
    plt.show()

    # And try to fit model again
    try_again = OdeFitter(my_ode, simulated_data.y, my_time)
    second_fit = try_again.fit()
    # logging.info(fit_report(second_fit))
    print(fit_report(second_fit))
    # This time the parameters get quite close ...
    assert True


def test_identifiability_with_external_temp(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries):
    # Fit model to data, then simulate data from that model (with more time points)
    # and try to fit a new model again. This is done to test if the model is identifiable.
    n_clusters = 4
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    my_time_series_expressions.keep_only_de_genes(std_cutoff=1.5)
    my_time, my_data = \
        my_time_series_expressions.get_clusters_expressions_with_time(n_clusters)

    for i, row in enumerate(my_data[:, ]):
        plt.plot(my_time, row, label=f"Real{i}", color=sns.color_palette()[i],
                 marker='o', linestyle='')

    initial_sim_fit = OdeFitter(my_ode, my_data, my_time)
    # Note: look into the initial parameter values
    optimal_fit = initial_sim_fit.fit()
    logging.info(fit_report(optimal_fit))
    # print(fit_report(optimal_fit))

    # Next step: simulate the data with these params and more data points
    more_time = np.arange(10, 1400, 10)
    simulated_data = initial_sim_fit.predict_values(optimal_fit.params, more_time)
    for i, row in enumerate(simulated_data.y):
        plt.plot(simulated_data.t, row, label=f'Fitted{i}',
                 color=sns.color_palette()[i], linestyle='dashed')
    plt.legend()
    plt.show()

    # And try to fit model again
    try_again = OdeFitter(my_ode, simulated_data.y, simulated_data.t)
    second_fit = try_again.fit()
    for i, row in enumerate(simulated_data.y):
        plt.plot(simulated_data.t, row, label=f'Simulated{i}',
                 color=sns.color_palette()[i], linestyle='dashed')

    second_sim = try_again.predict_values(second_fit.params, simulated_data.t)
    for i, row in enumerate(second_sim.y):
        plt.plot(second_sim.t, row, label=f'Fitting to simulated{i}',
                 color=sns.color_palette()[i])
    plt.legend()
    plt.show()

    logging.info(fit_report(second_fit))
    # print(fit_report(second_fit))
    assert True