import csv
import logging
from pathlib import Path

import pandas as pd

from Expressions.ExpressionMatrix import ExpressionMatrixTraining, \
    ExpressionMatrixTimeSeries, ExpressionMatrix
from helpers import plot_y_and_y_hat

logging.basicConfig(level=logging.INFO)

def test_get_cluster_per_gene(static_expression_two_temps_arabidopsis):
    """For each gene, check to which cluster it belongs"""
    n_clusters = 5
    static_expression_two_temps_arabidopsis.keep_only_wt_samples()
    static_expression_two_temps_arabidopsis.keep_genes_above_deviation_cutoff()
    # Needs to be training because we do clustering:
    train_expr_mat = static_expression_two_temps_arabidopsis.to_expressionmatrix_training()
    train_expr_mat.do_hierachical_clustering(n_clusters)
    some_clustering_dict = train_expr_mat.get_genes_per_cluster()
    assert len(some_clustering_dict) == n_clusters


def test_time_series_expressions(my_time_series_expressions: ExpressionMatrixTimeSeries):
    """Check if we can plot module expressions over time"""
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.keep_genes_above_deviation_cutoff(cutoff=1.5)
    my_time_series_expressions.do_hierachical_clustering(4)
    my_time_series_expressions.plot_clusters_over_time()
    assert True

def test_keep_n_most_deviating_genes(my_time_series_expressions: ExpressionMatrixTimeSeries):
    """Check if we can plot module expressions over time"""
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    my_time_series_expressions.keep_n_most_deviating_genes(1000, method='mad')
    my_time, my_data = my_time_series_expressions.get_clusters_expressions_with_time(6)
    plot_y_and_y_hat(y_real=my_data, t_real=my_time)

    assert True

def test_plot_non_tfs(my_time_series_expressions: ExpressionMatrixTimeSeries):
    """See if we can split off transcription factors, and plot their expression"""
    path_to_tfdb_file = Path(
        '../data/resources/Ath_TF_list.txt')
    non_tfs, tfs = my_time_series_expressions.split_off_tfs(path_to_tfdb_file)
    non_tfs.keep_genes_above_deviation_cutoff(cutoff=1.8)
    non_tfs.do_hierachical_clustering(4)
    non_tfs.plot_clusters_over_time()
    assert True


def test_get_lpan_input(my_time_series_expressions: ExpressionMatrixTimeSeries):
    """Check if we can extract modules from data, and save data so it can be
    used by LPAN and other pipelines downstream. Creates a file with expressions
    that LPAN uses directly, also creates file which contains edges between
    TFs and the module they belong to."""
    nr_clusters = 4
    some_cutoff = 1.5
    path_to_tfdb_file = Path(
        '../data/resources/Ath_TF_list.txt')
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    _, tfs = my_time_series_expressions.split_off_tfs(path_to_tfdb_file)
    # Only get DE genes
    my_time_series_expressions.keep_genes_above_deviation_cutoff(cutoff=some_cutoff)
    tfs.keep_genes_above_deviation_cutoff(cutoff=some_cutoff)

    # For all genes, convert into lpan format
    lpan_input_non_tf = my_time_series_expressions.get_lpan_input_modules(
        n_clusters=nr_clusters)
    # For TFs, convert into lpan format
    lpan_input_tf = tfs.get_lpan_input_tfs()
    # Merge and save to file
    lpan_input_everything = pd.concat([lpan_input_tf, lpan_input_non_tf])
    lpan_input_everything.to_csv(Path('../data/sample_file_for_lpan.csv'),
                                 quoting=csv.QUOTE_NONNUMERIC)

    out_file_path = Path(
        '../data/time_series_datasets/my_clustering_edgelist.csv')
    my_time_series_expressions.save_cluster_gene_edge_list(
        out_file_path=out_file_path, tf_filter_list=tfs.df.index.tolist())

    assert True
