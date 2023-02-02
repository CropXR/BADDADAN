import logging

from Expressions.ExpressionMatrix import ExpressionMatrix
from predict_from_static_expressions import plot_pred_vs_real, \
    fit_log_regress_model, infer_with_log_reg_model

def test_transfer_from_plant_to_seed(
        static_expression_two_temps_arabidopsis: ExpressionMatrix,
        my_seed_expressions: ExpressionMatrix):
    model, module_map = fit_log_regress_model(static_expression_two_temps_arabidopsis, n_cluster=10, std_cutoff=0.3)
    my_seed_expressions = my_seed_expressions.to_expressionmatrix_test()
    my_seed_expressions.quantile_normalize(static_expression_two_temps_arabidopsis)
    logging.info(f'{model.coef_=}')
    infer_with_log_reg_model(my_seed_expressions, model, module_map, use_random_classifier=False)
    # plot_pred_vs_real(static_expression_two_temps_arabidopsis, n_cluster)

    # my_seed_expressions.quantile_normalize(static_expression_two_temps_arabidopsis)
    # Initial implementation: first filter by overlapping genes

    print()


def test_random_classifier(
        static_expression_two_temps_arabidopsis: ExpressionMatrix,
        my_seed_expressions: ExpressionMatrix):
    _, module_map = fit_log_regress_model(
        static_expression_two_temps_arabidopsis, n_cluster=5, std_cutoff=.3)
    my_seed_expressions = my_seed_expressions.to_expressionmatrix_test()
    my_seed_expressions.quantile_normalize(static_expression_two_temps_arabidopsis)
    infer_with_log_reg_model(my_seed_expressions, None, module_map,
                             use_random_classifier=True)
    assert True