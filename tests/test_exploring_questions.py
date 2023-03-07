from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from exploring_questions import inference_with_thickened_ode_structure, \
    inference_with_thinned_ode_structure


def test_inference_with_wrong_ode_structure(my_tf2_ode: OdeModel,
                                            my_time_series_expressions: ExpressionMatrixTimeSeries):
    inference_with_thickened_ode_structure(my_tf2_ode)
    inference_with_thinned_ode_structure(my_tf2_ode)

