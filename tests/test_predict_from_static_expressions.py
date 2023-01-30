import logging

from Expressions.ExpressionMatrix import ExpressionMatrix
from predict_from_static_expressions import plot_pred_vs_real, \
    fit_log_regress_model, infer_with_log_reg_model

def test_transfer_from_plant_to_seed(
        static_expression_two_temps_arabidopsis: ExpressionMatrix,
        my_seed_expressions: ExpressionMatrix):
    # plant_genes = set(static_expression_two_temps_arabidopsis.get_gene_names())
    # seed_genes = set(my_seed_expressions.get_gene_names())
    # assert len(plant_genes & seed_genes) > 1
    model, module_map = fit_log_regress_model(static_expression_two_temps_arabidopsis)
    my_seed_expressions = my_seed_expressions.to_expressionmatrix_test()
    my_seed_expressions.quantile_normalize(static_expression_two_temps_arabidopsis)
    infer_with_log_reg_model(my_seed_expressions, model, module_map)
    # plot_pred_vs_real(static_expression_two_temps_arabidopsis, n_cluster)




    # my_seed_expressions.quantile_normalize(static_expression_two_temps_arabidopsis)
    # Initial implementation: first filter by overlapping genes

    print()
