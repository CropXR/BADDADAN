import logging

from matplotlib import pyplot as plt
import pandas as pd

from Expressions.ExpressionMatrix import ExpressionMatrix
from predict_from_static_expressions import plot_pred_vs_real, \
    fit_log_regress_model, infer_with_log_reg_model, \
    compare_expression_of_clusters_in_seed_vs_plant, compare_pca, \
    compare_expression_distributions


def test_transfer_from_plant_to_seed(
        static_expression_two_temps_arabidopsis: ExpressionMatrix,
        my_seed_expressions: ExpressionMatrix):
    static_expression_two_temps_arabidopsis = \
        static_expression_two_temps_arabidopsis.to_expressionmatrix_training()
    static_expression_two_temps_arabidopsis.keep_only_wt_samples()
    model, module_map = fit_log_regress_model(static_expression_two_temps_arabidopsis, n_cluster=2, std_cutoff=0.3)
    my_seed_expressions = my_seed_expressions.to_expressionmatrix_test()
    my_seed_expressions.quantile_normalize(static_expression_two_temps_arabidopsis)
    logging.info(f'{model.coef_=}')
    infer_with_log_reg_model(my_seed_expressions, model, module_map)
    # plot_pred_vs_real(static_expression_two_temps_arabidopsis, n_cluster)

    # my_seed_expressions.quantile_normalize(static_expression_two_temps_arabidopsis)
    # Initial implementation: first filter by overlapping genes

    print()


def test_random_classifier(
        static_expression_two_temps_arabidopsis: ExpressionMatrix,
        my_seed_expressions: ExpressionMatrix):
    n_cluster = 10
    static_expression_two_temps_arabidopsis = static_expression_two_temps_arabidopsis.to_expressionmatrix_training()
    static_expression_two_temps_arabidopsis.keep_only_wt_samples()
    my_seed_expressions = my_seed_expressions.to_expressionmatrix_test()
    all_accuracies = []
    all_roc = []
    for _ in range(1):
        random_model, module_map = fit_log_regress_model(
            static_expression_two_temps_arabidopsis, n_cluster=n_cluster,
            std_cutoff=.3, random_clustering=False)
        my_seed_expressions.quantile_normalize(static_expression_two_temps_arabidopsis)

        acc, roc_score = infer_with_log_reg_model(my_seed_expressions,
                                     random_model, module_map, do_plotting=False
                                     )
        all_accuracies.append(acc)
        all_roc.append(roc_score)

    print('Acc')
    print(pd.Series(all_accuracies).describe())
    print('ROC')
    print(pd.Series(all_roc).describe())
    assert True


def test_expression_of_clusters_in_seeds_vs_plants(
        static_expression_two_temps_arabidopsis: ExpressionMatrix,
        my_seed_expressions: ExpressionMatrix):
    static_expression_two_temps_arabidopsis.keep_only_de_genes(1)
    my_seed_expressions = my_seed_expressions.to_expressionmatrix_test()
    my_seed_expressions.quantile_normalize(
        static_expression_two_temps_arabidopsis)
    static_expression_two_temps_arabidopsis.quantile_normalize()
    compare_expression_distributions(my_seed_expressions,
                                     static_expression_two_temps_arabidopsis)
