import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from ExpressionMatrix import ExpressionMatrixTraining, \
    ExpressionMatrixTimeSeries, ExpressionMatrix


def test_get_cluster_per_gene(my_expression_matrix: ExpressionMatrix):
    """For each gene, check to which cluster it belongs"""
    n_clusters = 5
    my_expression_matrix = my_expression_matrix.remove_non_wt()
    my_expression_matrix.keep_only_de_genes()
    # Needs to be training because we do clustering:
    train_expr_mat = ExpressionMatrixTraining(my_expression_matrix.df)
    train_expr_mat.do_hierachical_clustering(n_clusters)
    some_clustering_dict = train_expr_mat.get_genes_per_cluster()
    assert len(some_clustering_dict) == n_clusters


def test_time_series_expressions(my_time_series_expressions: ExpressionMatrixTimeSeries):
    """Check if we can plot module expressions over time"""
    my_time_series_expressions.get_only_shoot()
    my_time_series_expressions.keep_only_de_genes(std_cutoff=1.5)
    expressions_array = my_time_series_expressions.plot_clusters_over_time(8)
    assert True


def test_plot_non_tfs(my_time_series_expressions: ExpressionMatrixTimeSeries):
    """See if we can split off transcription factors, and plot their expression"""
    path_to_tfdb_file = Path(
        '/home/bnoordijk/phd/sandbox_gene_expression/Ath_TF_list.txt')
    non_tfs, tfs = my_time_series_expressions.split_off_tfs(path_to_tfdb_file)
    non_tfs.keep_only_de_genes(std_cutoff=1.8)
    non_tfs.plot_clusters_over_time(4)
    assert True


def test_get_lpan_input(my_time_series_expressions: ExpressionMatrixTimeSeries):
    """Check if we can extract modules from data, and save data so it can be
    used by LPAN and other pipelines downstream. Creates a file with expressions that LPAN uses directly, and also contains a file which maps """
    nr_clusters = 40
    some_cutoff = 1.5
    path_to_tfdb_file = Path(
        '/home/bnoordijk/phd/sandbox_gene_expression/Ath_TF_list.txt')
    my_time_series_expressions.get_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    _, tfs = my_time_series_expressions.split_off_tfs(path_to_tfdb_file)
    my_time_series_expressions.keep_only_de_genes(std_cutoff=some_cutoff)
    tfs.keep_only_de_genes(std_cutoff=some_cutoff)
    lpan_input_non_tf = my_time_series_expressions.get_lpan_input(
        n_clusters=nr_clusters, index_prefix='MODULE')

    with Path('../data/my_clustering_dict.json').open('w') as f:
        f.write(json.dumps(my_time_series_expressions.get_cluster_per_gene()))

    lpan_input_tf = tfs.get_lpan_input(n_clusters=None, index_prefix='TF_')

    lpan_input_everything = pd.concat([lpan_input_tf, lpan_input_non_tf])
    lpan_input_everything.to_csv(Path('../data/sample_lpan_output.csv'),
                                 quoting=csv.QUOTE_NONNUMERIC)
    assert True
