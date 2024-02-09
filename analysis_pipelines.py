import copy
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries, \
    AggregationMethod
from helpers import get_info_from_gse65046
# from main import fit_ode_to_two_datasets


def pipeline_from_atted_clustering(experiment_path: Path,
                                   soft_file_path: Path,
                                   atted_linkage_matrix: Path,
                                   atted_path: Path,
                                   metabolites_path: Path,
                                   edge_cor_threshold: float,
                                   nr_clusters: int,
                                   top_nr_clusters: int,
                                   do_log2: bool,
                                   agg_method: AggregationMethod) \
        -> (ExpressionMatrixTimeSeries, ModuleRegulatoryNetwork):
    expr_mat_time: ExpressionMatrixTimeSeries = ExpressionMatrixTimeSeries.from_geo_file(
        soft_file_path,
        log2_transform=do_log2,
        annotate_from_gpl=True
    )
    expr_mat_time.column_parser = get_info_from_gse65046
    expr_mat_time.summary_method = agg_method

    aba_series = parse_metabolite_data(experiment_path, metabolites_path)

    expr_mat_time.add_phenotypes({'aba': aba_series})

    expr_mat_time.assign_clusters_from_linkage_matrix(atted_linkage_matrix,
                                                      nr_clusters,
                                                      atted_path=atted_path)
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
                                          remove_low_corr=True)
    my_grn.set_up_or_downregulation(expr_mat_time, do_plotting=False,
                                    threshold=threshold)
    # my_grn.plot_network(nx.d  raw_kamada_kawai, with_labels=False)
    module_module = my_grn.get_module_module_network()
    # # module_module.graph = nx.create_empty_copy(module_module.graph, with_data=False)
    module_module.plot_network(with_labels=True, out_path=module_plot_path)
    return module_module
