import csv
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
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
    my_time_series_expressions.do_hierachical_clustering(4)
    my_time_series_expressions.plot_clusters_over_time()
    assert True

def test_parse_65046_soft_file():
    soft_path = Path('../data/gse65046/GSE65046_family.soft')
    annotation_path = Path('../data/resources/96_plates.csv')
    # annotation = ExpressionArrayAnnotation(annotation_path,
    #                                        sep=',',
    #                                        array_type='catma')
    expressions = ExpressionMatrixTimeSeries.from_geo_file(soft_path,
                                                    log2_transform=True)
    print()
    assert expressions