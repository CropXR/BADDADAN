import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import dill as pickle
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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
