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


def test_assign_clusters_from_linkage_matrix(my_time_series_expressions):

    print()
    original_column_count = len(my_time_series_expressions.df.columns)
    my_time_series_expressions.merge_biological_samples()
    new_column_count = len(my_time_series_expressions.df.columns)
    assert new_column_count < original_column_count
    my_time_series_expressions.assign_clusters_from_linkage_matrix(
        linkage_matrix_path=Path('../data/preprocessed_data/summed_atted_gse65046/summed_dist_no_negative_dists/summed_distances_complete_linkage.npy'),
        nr_clusters=2,
        atted_path=Path('../data/preprocessed_data/summed_atted_gse65046/summed_dist_no_negative_dists/atted_local_dist_summed_no_negative.parquet.gzip')
    )
    assert my_time_series_expressions.has_been_clustered

















#
# def test_get_cluster_per_gene(static_expression_two_temps_arabidopsis):
#     """For each gene, check to which cluster it belongs"""
#     n_clusters = 5
#     static_expression_two_temps_arabidopsis.keep_only_wt_samples()
#     static_expression_two_temps_arabidopsis.keep_genes_above_deviation_cutoff()
#     # Needs to be training because we do clustering:
#     train_expr_mat = static_expression_two_temps_arabidopsis.to_expressionmatrix_training()
#     train_expr_mat.do_hierachical_clustering(n_clusters)
#     some_clustering_dict = train_expr_mat.get_genes_per_cluster()
#     assert len(some_clustering_dict) == n_clusters
#
#
# def test_time_series_expressions(
#         my_time_series_expressions: ExpressionMatrixTimeSeries):
#     """Check if we can plot module expressions over time"""
#     my_time_series_expressions.keep_only_shoot()
#     my_time_series_expressions.keep_genes_above_deviation_cutoff(cutoff=1.5)
#     my_time_series_expressions.do_hierachical_clustering(4)
#     my_time_series_expressions.plot_clusters_over_time()
#     assert True
#
#
# def test_keep_n_most_deviating_genes(
#         my_time_series_expressions: ExpressionMatrixTimeSeries):
#     """Check if we can plot module expressions over time"""
#     my_time_series_expressions.keep_only_shoot()
#     my_time_series_expressions.merge_biological_samples()
#     my_time_series_expressions.keep_n_most_deviating_genes(1000, method='mad')
#     my_time_series_expressions.do_hierachical_clustering(4)
#     my_time_series_expressions.plot_clusters_over_time()
#     assert True
#
#
# def test_do_flame_clustering(
#         my_time_series_expressions: ExpressionMatrixTimeSeries):
#     my_time_series_expressions.keep_only_shoot()
#     my_time_series_expressions.merge_biological_samples()
#     my_time_series_expressions.keep_n_most_deviating_genes(150, method='mad')
#     # TODO properly get transcription factors
#     my_time_series_expressions.do_flame_clustering(
#         Path('../bins/flame_clustering'))
#     out_file_path = Path(
#         '../data/garbage/my_clustering_edgelist.csv')
#     my_time_series_expressions.save_tf_produced_by_module_file(
#         out_file_path=out_file_path)
#     my_time_series_expressions.plot_clusters_over_time()
#     my_time_series_expressions.write_tf2_input_file(
#         Path('../data/garbage/flame_test_cluster.txt'))
#
#     my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(
#         Path('../data/garbage/02_tf2network_output.tsv'))
#     my_grn.plot_network(nx.draw)
#     my_grn.add_tf_module_mappings(
#         Path('../data/garbage/my_clustering_edgelist.csv'))
#     my_grn.clean_up_network()
#     my_grn.plot_network()
#
#
# def test_get_lpan_input(
#         my_time_series_expressions: ExpressionMatrixTimeSeries):
#     """Check if we can extract modules from data, and save data so it can be
#     used by LPAN and other pipelines downstream. Creates a file with expressions
#     that LPAN uses directly, also creates file which contains edges between
#     TFs and the module they belong to."""
#     raise NotImplementedError("Don't use this anymore")
#     nr_clusters = 4
#     some_cutoff = 1.5
#     path_to_tfdb_file = Path(
#         '../data/resources/Ath_TF_list.txt')
#     my_time_series_expressions.keep_only_shoot()
#     my_time_series_expressions.merge_biological_samples()
#     _, tfs = my_time_series_expressions.split_off_tfs(path_to_tfdb_file)
#     # Only get DE genes
#     my_time_series_expressions.keep_genes_above_deviation_cutoff(
#         cutoff=some_cutoff)
#     tfs.keep_genes_above_deviation_cutoff(cutoff=some_cutoff)
#
#     # For all genes, convert into lpan format
#     lpan_input_non_tf = my_time_series_expressions.get_lpan_input_modules(
#         n_clusters=nr_clusters)
#     # For TFs, convert into lpan format
#     lpan_input_tf = tfs.get_lpan_input_tfs()
#     # Merge and save to file
#     lpan_input_everything = pd.concat([lpan_input_tf, lpan_input_non_tf])
#     lpan_input_everything.to_csv(Path('../data/sample_file_for_lpan.csv'),
#                                  quoting=csv.QUOTE_NONNUMERIC)
#
#     out_file_path = Path(
#         '../data/time_series_datasets/my_clustering_edgelist.csv')
#     my_time_series_expressions.save_tf_produced_by_module_file(
#         out_file_path=out_file_path, tf_list_path=tfs.df.index.tolist())
#
#     assert True
#
#
# def test_parse_65046_soft_file():
#     soft_path = Path(
#         '../data/raw_data/expression_datasets/gse65046/GSE65046_family.soft')
#     annotation_path = Path('../data/resources/96_plates.csv')
#     # annotation = ExpressionArrayAnnotation(annotation_path,
#     #                                        sep=',',
#     #                                        array_type='catma')
#     expressions = ExpressionMatrixTimeSeries.from_geo_file(soft_path,
#                                                            log2_transform=True)
#     print()
#     assert expressions
#
#
