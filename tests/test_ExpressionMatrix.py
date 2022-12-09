from pathlib import Path

import pandas as pd
import pytest

from ExpressionMatrix import ExpressionMatrixTraining


def test_get_cluster_per_gene(my_expression_matrix):
    n_clusters = 5
    my_expression_matrix = my_expression_matrix.remove_non_wt()
    my_expression_matrix = my_expression_matrix.get_only_de_genes()
    # Needs to be training because we do clustering:
    train_expr_mat = ExpressionMatrixTraining(my_expression_matrix.df)
    train_expr_mat.do_hierachical_clustering(n_clusters, inplace=True)
    some_clustering_dict = train_expr_mat.get_genes_per_cluster()
    assert len(some_clustering_dict) == n_clusters


def test_time_series_expressions(my_time_series_expressions):
    my_time_series_expressions.get_only_shoot(inplace=True)
    my_time_series_expressions.get_only_de_genes(std_cutoff=1.5, inplace=True)
    expressions_array = my_time_series_expressions.clusters_over_time(8)

    assert True
