import logging

import numpy as np
import pytest
from lmfit import fit_report
from matplotlib import pyplot as plt

from ExpressionMatrix import ExpressionMatrixTimeSeries
from ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from OdeFitter import OdeFitter
from OdeInference import OdeInference


def test_convert_to_ode(my_ode: OdeInference):
    logging.info(f'{my_ode=}')
    params = np.random.rand(my_ode.nr_params).tolist()
    dydt = my_ode(None, [0, 1, 0, 0], *params)
    logging.info(f'{dydt=}')
    assert dydt != [0, 1, 0, 0]


def test_fit_ode_to_data(
        my_ode: OdeInference,
        my_time_series_expressions: ExpressionMatrixTimeSeries):
    n_clusters = 4
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    my_time_series_expressions.keep_only_de_genes(std_cutoff=1.5)
    # TODO check if my_data is correct format and not inverted or something
    my_time, my_data = \
        my_time_series_expressions.get_clusters_expressions_with_time(n_clusters)
    fitter = OdeFitter(my_ode, my_data, my_time)
    # Note: look into the initial parameter values
    optimal_fit = fitter.fit()
    logging.info(fit_report(optimal_fit))
    assert True


def test_identifiability(
        my_ode: OdeInference,
        my_time_series_expressions: ExpressionMatrixTimeSeries
):
    """
    Simulate some data, and then try to fit an ODE to it to test identifiability.
    """
    my_time = np.array([15., 30., 60., 180., 240., 360., 720., 1440.])
    my_time = np.linspace(0, 1000, 1000)
    generator = OdeFitter(my_ode, measured_data=np.random.rand(4, 1), time_points=my_time)
    # generator.set_params({})
    my_params = generator.params
    print('Modify that shit now')
    my_params['beta3'].value = 0.1
    my_params['beta0'].value = my_params['beta1'].value = 0.05
    my_params['beta2'].value = my_params['beta4'].value = -0.05
    for i in [0, 1, 2, 4]:
        my_params[f'd{i}'].value = 0.01
    sim_data = generator.predict_values(my_params,
                                        generator.time_points,
                                        generator.odes)
    # for row in sim_data.y:
    #     plt.plot(sim_data.t, row)

    fitter = OdeFitter(my_ode, sim_data.y, sim_data.t)
    # Note: look into the initial parameter values
    fit_result = fitter.fit()
    # TODO compare this to actual data and incorporate inhibition mechanisms?
    logging.info(fit_report(fit_result))
    logging.info(f'{my_params=}')


    optimal_fit = fitter.predict_values(fit_result.params, sim_data.t, my_ode)
    colours = ['b', 'r', 'g', 'y']
    for i in range(4):
        colour = colours[i]
        true = sim_data.y[i, :]
        fit = optimal_fit.y[i, :]
        plt.plot(my_time, true, '-', color=colour)
        plt.plot(my_time, fit, '--', color=colour)
    plt.show()
    assert True

