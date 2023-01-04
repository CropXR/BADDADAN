import logging

import numpy as np
import pytest

from ExpressionMatrix import ExpressionMatrixTimeSeries
from ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from OdeFitter import OdeFitter
from OdeInference import OdeInference


def test_convert_to_ode(my_ode: OdeInference):
    logging.info(f'{my_ode=}')
    params = np.random.rand(my_ode.nr_params).tolist()
    dydt = my_ode(None, [0, 1, 0, 0], params)
    logging.info(f'{dydt=}')
    assert dydt != [0, 1, 0, 0]

def test_fit_ode_to_data(
        my_ode: OdeInference,
        my_time_series_expressions: ExpressionMatrixTimeSeries):
    n_clusters = 4
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    my_time_series_expressions.keep_only_de_genes(std_cutoff=1.5)
    my_time, my_data = my_time_series_expressions.get_clusters_expressions_with_time(n_clusters)
    fitter = OdeFitter(my_ode, my_data, my_time)
    fitter.fit(t_end=180)
