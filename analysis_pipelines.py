import copy
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import dill as pickle
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from goatools.anno.idtogos_reader import IdToGosReader
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.utils import read_geneset
from matplotlib import pyplot as plt
from tqdm import tqdm
from goatools.obo_parser import GODag


from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries, \
    AggregationMethod
from data_wrangling import expr_mat_from_drought, expr_mat_from_emexp, \
    merge_ath_annotation_for_goatools
from helpers import get_info_from_gse65046

@dataclass
class ClusteringArgs:
    """Arguments to use for clustering with different linkage matrices"""
    input_dist_name: str
    linkage_matrix_path: Path | None
    original_dist_path: Path | None

def compare_clusterings_for_ode_use(expr_mat_time: ExpressionMatrixTimeSeries,
                                    experiment_path: Path,
                                    summed_linkage_matrix: Path,
                                    atted_linkage_matrix: Path,
                                    atted_path: Path,
                                    summed_dist_matrix_path: Path,
                                    nr_clusters: int,
                                    edge_cor_threshold: float,
                                    top_nr_clusters: int,
                                    tf2_in_name: str,
                                    tf2_out_name: str):

    clustering_arg_list = [
        ClusteringArgs('atted_only', atted_linkage_matrix, atted_path),
        ClusteringArgs('summed_dists', summed_linkage_matrix, summed_dist_matrix_path),
        ClusteringArgs('local_dists', None, None),
    ]

    out_dict = {}
    for clustering_arg in clustering_arg_list:
        logging.info(f'{clustering_arg.input_dist_name} analysis now')

        expr_mat_time_copy = copy.deepcopy(expr_mat_time)
        if clustering_arg.input_dist_name == 'local_dists':
            expr_mat_time_copy.do_hierachical_clustering(nr_clusters)
        else:
            expr_mat_time_copy.assign_clusters_from_linkage_matrix(
                clustering_arg.linkage_matrix_path,
                nr_clusters,
                distance_matrix_path=clustering_arg.original_dist_path
            )
    #     expr_mat_time_copy.write_tf2_input_file(
    #         experiment_path / f'01_{clustering_arg.input_dist_name}_tf2input.txt')
        out_dict[clustering_arg.input_dist_name] = expr_mat_time_copy

    # print()
    with open(experiment_path / 'all_clusterings_expr_mat_dict.pkl',
              'wb') as f:
        pickle.dump(out_dict, f)

    with open(experiment_path / 'all_clusterings_expr_mat_dict.pkl',
              'rb') as f:
        out_dict: Dict[str: ExpressionMatrixTimeSeries] = pickle.load(f)

    # # Check if 56 modules with >0 TF in local dists have significant overlap
    # compare_modules_to_local_modules_with_tfbs(
    #     out_dict,
    #     experiment_path / f'02_local_dists_tf2network_output.tsv')

    explained_var_df_list = []
    tf_prod_df_list = []
    tf_consistency_list = []
    nr_tfs_between_modules_list = []
    coex_score_list = []
    for dist_name, expression_df in out_dict.items():
        # expr_mat_time_copy.check_enrichment_string_db()
        tf2_in_file = experiment_path / f'01_{dist_name}_tf2input.txt'
        tf2_out_file = experiment_path / f'02_{dist_name}_tf2network_output.tsv'

        # Get coherence for each cluster
        explained_vars = expression_df.get_all_explained_vars()
        sizes = expression_df.get_module_sizes()
        explained_vars = explained_vars.to_frame(name='explained_var')
        explained_vars['input_dists'] = dist_name
        explained_vars['size'] = sizes
        stdevs = expression_df.get_std_per_cluster(mean_over_all_samples=True)
        explained_vars['stdev'] = stdevs
        explained_var_df_list.append(explained_vars)

        # # Get TF2 Coexpression score for each cluster
        # coex_score = get_coex_from_tf2_output(tf2_out_file)
        # coex_score = coex_score.to_frame(name='coexpression_scores')
        # coex_score['input_dists'] = dist_name
        # coex_score_list.append(coex_score)
        #
        # # # For now just do this on all modules right?
        # my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(tf2_out_file)
        # my_grn.add_tf_module_mappings(tf2_in_file,
        #                               from_tf2_input=True)
        # my_grn.clean_up_network()
        # my_grn.set_up_or_downregulation(expression_df,
        #                                 threshold=0,
        #                                 do_plotting=False)
        # my_grn.get_intermodular_connection_df()
        #
        # size_distribution = my_grn.see_how_many_tfs_between_modules()
        # size_distribution['method'] = dist_name
        # nr_tfs_between_modules_list.append(size_distribution)
        # consistency_at_threshold = my_grn.check_consistency_between_module_regulations()
        # consistency_at_threshold.name = dist_name
        # tf_consistency_list.append(consistency_at_threshold)

    vars_df = pd.concat(explained_var_df_list)
    # sns.scatterplot(data=vars_df, x='size', y='explained_var', hue='stdev')
    # plt.xscale('log')
    # plt.show()

    sns.jointplot(data=vars_df, x='size', y='explained_var', hue='input_dists')
    plt.xscale('log')
    plt.savefig(experiment_path / 'explained_var_jointplot.svg')
    plt.close()

    sns.scatterplot(data=vars_df, x='stdev',
                    y='explained_var',
                    hue='size',
                    style='input_dists')
    plt.savefig(experiment_path / 'explained_var_scatterplot.svg')
    plt.close()

    sns.violinplot(data=vars_df, y='explained_var', x='input_dists')
    plt.savefig(experiment_path / 'explained_var_violinplot.svg')
    plt.close()

    return

    coex_df = pd.concat(coex_score_list)
    sns.violinplot(data=coex_df, y='coexpression_scores', x='input_dists')
    plt.show()

    nr_tfs_between_mods = pd.concat(nr_tfs_between_modules_list)
    sns.set_style('white')
    sns.histplot(data=nr_tfs_between_mods, x='nr_tfs_between_modules',
                 hue='method', stat='percent',
                 discrete=True, common_norm=False, multiple='dodge',
                 shrink=.8)
    sns.despine()
    plt.show()

    consistent_tf_df = pd.concat(tf_consistency_list, axis=1)
    melted_consistent_tf_df = consistent_tf_df.reset_index().melt(
        id_vars='index',
        value_name='% agree',
        var_name='input dists'
    )
    melted_consistent_tf_df = melted_consistent_tf_df.rename(
        columns={'index': 'cutoff'})
    sns.lineplot(data=melted_consistent_tf_df,
                 x='cutoff',
                 y='% agree',
                 hue='input dists')
    plt.show()



    tf_prod_df = pd.concat(tf_prod_df_list)
    sns.boxplot(data=tf_prod_df, y='tf_prod_module_cor', x='input_dists')
    plt.show()

    plt.savefig(experiment_path / 'tf_prod_module_cor_boxplot.svg')
    plt.close()


def assign_clusters_and_infer_intermodular_network(
        experiment_path: Path, expr_mat_time,
        summed_linkage_matrix: Path,
        summed_dist_matrix_path: Path,
        nr_clusters: int,
        edge_cor_threshold: float,
        top_nr_clusters: int,
        tf2_in_name: str,
        tf2_out_name: str):
    """Cluster expression matrix, and infer intermodular network"""

    expr_mat_time.assign_clusters_from_linkage_matrix(
        summed_linkage_matrix,
        nr_clusters,
        distance_matrix_path=summed_dist_matrix_path)

    # sns.clustermap(expr_mat_time.df.iloc[:, :-1],
    #               row_linkage=np.load(summed_linkage_matrix),
    #               z_score=0)
    # plt.show()
    #
    # sns.clustermap(
    #     expr_mat_time.get_distance_matrix(),
    #     row_linkage=np.load('data/experiments/12_drought_data_with_gene_prefilter_and_zscore_proper/local_distances_complete_linkage.npy'),
    #     col_linkage=np.load('data/experiments/12_drought_data_with_gene_prefilter_and_zscore_proper/local_distances_complete_linkage.npy'),)
    # plt.show()
    #
    #
    # sns.clustermap(pd.read_parquet(summed_dist_matrix_path))
    # plt.show()

    return infer_intermodular_network(expr_mat_time, experiment_path,
                                      tf2_in_name, tf2_out_name,
                                      top_nr_clusters, edge_cor_threshold)


def infer_intermodular_network(expr_mat_time, experiment_path, tf2_in_name,
                               tf2_out_name, top_nr_clusters,
                               edge_cor_threshold):
    expr_mat_time.plot_sample_gene_heatmap()
    # explained_vars = expr_mat_time.get_all_explained_vars()
    expr_mat_time.plot_cluster_sizes(experiment_path / 'cluster_sizes.png')
    tf2_in_path = experiment_path / tf2_in_name
    tf2_out_path = experiment_path / tf2_out_name
    if not tf2_out_path.exists():
        expr_mat_time.write_tf2_input_file(tf2_in_path)
        expr_mat_time.post_to_tf2network(tf2_in_path, tf2_out_path)
    # expr_mat_time.assign_clusters_from_tf2_input(tf2_in_path, overwrite=False)
    expr_mat_time.keep_highest_z_clusters(top_nr_clusters,
                                          tf2_out_path,
                                          plotting_path=experiment_path)
    # expr_mat_time.plot_clusters_over_time()
    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=edge_cor_threshold,
        module_plot_path=experiment_path / 'global_cluster_module_network.svg')
    expr_mat_time.keep_only_modules_in_network(module_module)
    expr_mat_time.plot_clusters_over_time(
        out_path=experiment_path / 'global_cluster_expressions.svg',
        timescale='hours')
    return expr_mat_time, module_module


def explore_emtab_375(experiment_path: Path, in_file_path: Path,
                      summed_linkage_matrix: Path,
                      summed_dist_matrix_path: Path, nr_clusters: int,
                      do_log2: bool, agg_method: AggregationMethod,
                      gpl_path: str = None):

    if in_file_path.suffix == '.csv':
        expr_mat_time = expr_mat_from_emexp(in_file_path, agg_method, do_log2,
                                            gpl_path)
    else:
        raise NotImplementedError

    # local_path = experiment_path / 'local_dists.pkl'
    # expr_mat_time.save_distance_matrix(local_path)
    # sum_local_distance_and_atted(local_path, atted_path, experiment_path)
    expr_mat_time.assign_clusters_from_linkage_matrix(summed_linkage_matrix,
                                                      nr_clusters,
                                                      distance_matrix_path=summed_dist_matrix_path)
    # expr_mat_time.keep_n_most_deviating_genes(200)
    expr_mat_time.do_hierachical_clustering(nr_clusters)
    # TODO TF2Network enrichment

    expr_mat_time.plot_cluster_sizes()
    expr_mat_time.keep_highest_z_clusters(4, None)
    expr_mat_time.plot_clusters_over_time()
    plt.show()
    # tf2_in_path = experiment_path / '01_tf2_input.txt'
    # tf2_out_path = experiment_path / '02_tf2network_output.tsv'
    # expr_mat_time.post_to_tf2network(tf2_in_path, tf2_out_path)
    # # expr_mat_time.write_tf2_input_file(tf2_in_path)
    # print()

def pipeline_from_atted_clustering(experiment_path: Path, soft_file_path: Path,
                                   atted_linkage_matrix: Path,
                                   atted_path: Path, edge_cor_threshold: float,
                                   nr_clusters: int, top_nr_clusters: int,
                                   do_log2: bool,
                                   agg_method: AggregationMethod) \
        -> (ExpressionMatrixTimeSeries, ModuleRegulatoryNetwork):
    expr_mat_time: ExpressionMatrixTimeSeries = ExpressionMatrixTimeSeries.from_csv(
        soft_file_path, log2_transform=do_log2)
    expr_mat_time.column_parser = get_info_from_gse65046
    expr_mat_time.summary_method = agg_method

    expr_mat_time.assign_clusters_from_linkage_matrix(atted_linkage_matrix,
                                                      nr_clusters,
                                                      distance_matrix_path=atted_path)
    expr_mat_time.merge_biological_samples()

    # expr_mat_time.keep_n_most_deviating_genes(50)
    # expr_mat_time.do_hierachical_clustering(nr_clusters)
    expr_mat_time.plot_cluster_sizes(experiment_path / 'cluster_sizes.png')

    tf2_in_path = experiment_path / '01_tf2_input.txt'
    expr_mat_time.write_tf2_input_file(tf2_in_path)
    # expr_mat_time._do_random_clustering(nr_clusters)
    # expr_mat_time.see_pairwise_cluster_correlations('Random')
    # expr_mat_time.do_hierachical_clustering(nr_clusters)
    # expr_mat_time.see_pairwise_cluster_correlations('Hierarchical on dataset')
    # expr_mat_time.assign_clusters_from_tf2_input(tf2_in_path, overwrite=True)
    # expr_mat_time.see_pairwise_cluster_correlations('Post-cluster')
    # expr_mat_time.plot_cluster_sizes()

    # TF2Output file should be here
    tf2_out_path =  experiment_path / '02_tf2network_output.tsv'
    # expr_mat_time.get_z_score_of_cluster_characteristics(tf2_out_path, plotting=True)
    expr_mat_time.keep_highest_z_clusters(top_nr_clusters, tf2_out_path)
    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=edge_cor_threshold,
        module_plot_path=experiment_path / 'global_cluster_module_network.svg')

    expr_mat_time.keep_only_modules_in_network(module_module)

    expr_mat_time.plot_clusters_over_time(split_by_condition=['control', 'drought'],
                                          out_path=experiment_path / 'global_cluster_expressions.svg')

    return expr_mat_time, module_module


def parse_metabolite_data(experiment_path: Path, metabolites_path: Path):
    metabolite_time_series = pd.read_excel(metabolites_path,
                                           index_col=0,
                                           header=[0, 1, 2]
                                           )
    # Convert the time level to pd.timedelta
    columns_as_frame = metabolite_time_series.columns.to_frame()
    columns_as_frame['time'] = pd.to_timedelta(columns_as_frame['time'])
    multi_index = pd.MultiIndex.from_frame(columns_as_frame)
    metabolite_time_series = metabolite_time_series.set_axis(multi_index,
                                                             axis=1)
    metabolite_time_series = metabolite_time_series.groupby(
        level=[0, 1], axis=1).mean()
    metabolite_time_series = metabolite_time_series.dropna()
    # Start with ABA? -> Yes
    aba_series = metabolite_time_series.loc['Abscisic acid (ABA) ', :]
    aba_series = aba_series.reset_index()
    aba_series['time_days'] = pd.to_timedelta(aba_series['time']).astype('timedelta64[D]')
    sns.lineplot(data=aba_series, x='time_days', y='Abscisic acid (ABA) ',
                 hue='condition')
    aba_series = aba_series.drop('time_days', axis=1)
    plt.savefig(experiment_path / 'aba_time_series.png')
    plt.close()
    return aba_series


def local_clustering_on_atted_clusters(clustering_of_clusters_threshold,
                                       edge_cor_threshold,
                                       experiment_path,
                                       expr_mat_time,
                                       top_nr_clusters):
    # expr_mat_time.see_pairwise_cluster_correlations('Pre-selection')
    expr_mat_time.merge_correlating_modules(
        cutoff=clustering_of_clusters_threshold,
        criterion_type='maxclust',
        criterion_start_value=top_nr_clusters,
        criterion_step=-1)
    # expr_mat_time.see_pairwise_cluster_correlations('Post-selection')
    # with open('data/gse65046/500_clusters_mean_ExpressionMatrix.pkl',
    #           'wb') as f:
    #     pickle.dump(expr_mat_time, f)
    expr_mat_drought = copy.deepcopy(expr_mat_time)
    expr_mat_drought.keep_only_samples_with_string('drought')
    expr_mat_control = copy.deepcopy(expr_mat_time)
    expr_mat_control.keep_only_samples_with_string('control')
    new_tf2_in = experiment_path / '03_metacluster_tf2_input.txt'
    expr_mat_time.write_tf2_input_file(new_tf2_in)
    expr_mat_time.plot_clusters_over_time(
        split_by_condition=['control', 'drought'],
        out_path=experiment_path / 'local_cluster_expressions.svg')
    new_tf2_out = experiment_path / '04_tf2network_output.tsv'
    new_module_module = module_network_from_tf2_output(
        expr_mat_time,
        new_tf2_in,
        new_tf2_out,
        threshold=edge_cor_threshold,
        module_plot_path=experiment_path / 'local_cluster_module_network.svg')
    return expr_mat_time, new_module_module


def module_network_from_tf2_output(expr_mat_time,
                                   tf2_in_path,
                                   tf2_out_path,
                                   threshold,
                                   module_plot_path):
    my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(tf2_out_path)
    my_grn.add_tf_module_mappings(tf2_in_path,
                                  from_tf2_input=True)
    my_grn.keep_only_modules_of_interest(expr_mat_time)
    my_grn.clean_up_network()
    my_grn.check_if_tfs_created_by_module(expr_mat_time,
                                          do_plotting=False,
                                          remove_low_corr=False,
                                          assert_correlated=False)
    my_grn.set_up_or_downregulation(expr_mat_time, do_plotting=False,
                                    threshold=threshold)
    # my_grn.plot_network(nx.d  raw_kamada_kawai, with_labels=False)
    module_module = my_grn.get_module_module_network()
    # # module_module.graph = nx.create_empty_copy(module_module.graph, with_data=False)
    module_module.plot_network(nx.draw_kamada_kawai , with_labels=True, out_path=module_plot_path)
    logging.info(list(module_module.graph.edges(data=True)))
    return module_module

def set_background_genes(
        in_path,
        out_path,
        gene_annotation_path: Path
            = Path('data/resources/go_annotations/ATH_GO_GOSLIM.txt')
):
    """"""
    #
    # annotation_df = merge_ath_annotation_for_goatools(gene_annotation_path)
    # background_genes_path = Path(out_path)
    # annotation_df.reset_index()['locus name'].to_csv(
    #     background_genes_path,
    #     sep='\t',
    #     header=False,
    #     index=False
    # )