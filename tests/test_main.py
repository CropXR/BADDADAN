
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from DynamicModels.OdeModel import OdeModel

from main import fit_ode_to_data


def test_fit_ode_to_data(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries):
    fit_ode_to_data(my_ode, my_time_series_expressions)
