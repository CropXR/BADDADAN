from copy import deepcopy
from pathlib import Path
from typing import Dict

import dill as pickle
import mlflow
import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt

from DynamicModels.OdeFitterMultipleDatasets import OdeFitterMultipleDatasets
from DynamicModels.OdeLocalParameters import OdeLocalParameters

from DynamicModels.OdeModel import OdeModel
from DynamicModels.helper_scripts_for_fitting import fit_multiple_fitters
from Expressions.ExpressionMatrix import AggregationMethod, \
    ExpressionMatrixTimeSeries
from analysis_pipelines import assign_clusters_and_infer_intermodular_network, \
    explore_emtab_375, compare_clusterings_for_ode_use, \
    module_network_from_tf2_output
from data_wrangling import expr_mat_from_emexp, expr_mat_from_drought

from exploring_questions import plot_module_size_distributions, \
    sum_local_distance_and_atted, similarity_matrices_local_and_atted
from helpers import get_info_from_emtab375


def prefilter_genes_experiment(experiment_path):

    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)

    expr_mat_time_drought = expr_mat_from_drought(
        data_params['limma_drought_out_path'],
        hyper_params['agg_method'],
        hyper_params['do_log2_drought'])

    expr_mat_time_heat = expr_mat_from_emexp(
        data_params['limma_heat_out_path'],
        hyper_params['agg_method'],
        hyper_params['do_log2_heat'],
        data_params['heat_gpl_path']
    )

    cv_list = []
    df_list = []


    expr_mat_time_heat
    #
    #
    # for expr_mat_time, condition_name in zip(
    #         [expr_mat_time_drought, expr_mat_time_heat],
    #         ['drought', 'heat']
    # ):
    #     expr_mat_time.scatterplot_of_two_per_gene_stats(
    #         'std', 'cond_rmsd',
    #         plotting_func=sns.jointplot,
    #         title = f'{condition_name} _std_rmsd_no cutoff ({len(expr_mat_time.df)} genes)',
    #         out_path=experiment_path /  f'{condition_name}_no_cutoff.png')
    #
    #     # expr_mat_time.scatterplot_of_two_per_gene_stats(
    #     #     'mean', 'std',
    #     #     plotting_func=sns.jointplot,
    #     #     title = f'{condition_name} no cutoff ({len(expr_mat_time.df)} genes)',
    #     #     out_path=experiment_path /  f'{condition_name}_no_cutoff.png')
    #
    #     for cutoff in [0.25, 0.5, 0.75]:
    #         temp_expr_mat = deepcopy(expr_mat_time)
    #         # std_series = temp_expr_mat.plot_per_gene_std()
    #         temp_expr_mat.keep_genes_above_percentile_score(
    #             cutoff,
    #             method='cond_rmsd')
    #         # temp_expr_mat.scatterplot_of_two_per_gene_stats(
    #         #     'mean', 'std',
    #         #     plotting_func=sns.jointplot,
    #         #     title=f'{condition_name} cutoff={cutoff} perc. ({len(temp_expr_mat.df)} genes)',
    #         #     out_path=experiment_path / f'{condition_name}_{cutoff}_cutoff.png'
    #         # )
    #         # mad_series = expr_mat_time.plot_per_gene_mad()
    #         # cv_serie = expr_mat_time._calculate_gene_variation('cv')
    #         # cv_list.append(cv_serie)
    #         #
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('std', 'qcd')
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('mean', 'qcd')
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('std', 'mad')
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('std', 'cv')
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('std', 'qcd')
    #         # Median should roughly be good cutoff?
    #         # Or only remove lower 25th percentile?
    #         # Look at distribution of MAD
    #
    #
    # # cv_list[0].name = 'drought'
    # # cv_list[1].name = 'heat'
    # # merged_df = pd.concat(cv_list, axis=1, join='inner')
    # # sns.histplot(merged_df)
    # # print()






def figure_2_pipeline(experiment_path):
    for folder in experiment_path.iterdir():
        if not folder.name.endswith('_data') or folder.name.startswith('drought'):
            continue
        data_params, hyper_params, experiment_params = config_preprocess(folder)

        robustness_csv_path = Path(data_params['robustness_csv'])
        robustness_df = pd.read_csv(robustness_csv_path)
        sns.violinplot(data=robustness_df, x='input_dists', y='robustness')
        plt.ylim([0, 0.35])
        dataset_name = robustness_csv_path.name.split('_')[0]
        plt.title(dataset_name.capitalize())
        plt.savefig(folder.parent / 'figures' / f'{dataset_name}_robustness_violinplot.svg')
        plt.close()

        if folder.name.startswith('drought'):
            expr_mat_time = expr_mat_from_drought(
                in_file_path=data_params['soft_path'],
                agg_method=hyper_params['agg_method'],
                do_log2=hyper_params['do_log2']
            )
        elif folder.name.startswith('heat'):
            expr_mat_time = expr_mat_from_emexp(
                in_path=data_params['soft_path'],
                agg_method=hyper_params['agg_method'],
                do_log2=hyper_params['do_log2'],
                gpl_path=data_params['gpl_path']
            )
        else:
            raise NotImplementedError

        compare_clusterings_for_ode_use(
            expr_mat_time,
            experiment_path=folder,
            summed_linkage_matrix=data_params['linkage_path'],
            atted_linkage_matrix=Path(data_params['atted_linkage_matrix']),
            atted_path=Path(data_params['atted_path']),
            summed_dist_matrix_path=Path(data_params['dist_matrix_path']),
            nr_clusters=hyper_params['nr_clusters'],
            edge_cor_threshold=None,
            top_nr_clusters=None,
            tf2_in_name=None,
            tf2_out_name=None)




def module_size_pipeline(experiment_path):
    for file in experiment_path.iterdir():
        if file.name.endswith('expr_mat_dict.pkl'):
            plot_module_size_distributions(file)
    with mlflow.start_run():
        for file in experiment_path.iterdir():
            mlflow.log_artifact(str(file))
            # if not file.suffix in ['.npy', '.pkl', '.gzip']:
            #     mlflow.log_artifact(str(file))

def drought_from_wgcna(experiment_path,
                       ):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    wgcna_module_assignment = data_params['wgcna_module_assignment_path']
    wgcna_eigengenes = data_params['wgcna_eigengenes']
    df_eigengenes = pd.read_csv(wgcna_eigengenes)
    # sns.lineplot(df_eigengenes)
    # plt.show()
    expr_mat_time: ExpressionMatrixTimeSeries = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])
    expr_mat_time.assign_clusters_from_wgcna(wgcna_module_assignment)
    expr_mat_time.plot_cluster_sizes()

    tf2_in_path =experiment_path / data_params['tf2_in_name']
    tf2_out_path = experiment_path / data_params['tf2_out_name']
    # Post to tf2network
    expr_mat_time.write_tf2_input_file(
        out_path=tf2_in_path)

    expr_mat_time.do_genewise_normalisation()
    expr_mat_time.keep_highest_z_clusters(
        5,
        tf2_output_path=tf2_out_path,
    plotting_path=experiment_path)

    expr_mat_time.plot_clusters_over_time()

    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=hyper_params['edge_corr_threshold'],
        module_plot_path=experiment_path / 'global_cluster_module_network.svg')

    expr_mat_time.keep_only_modules_in_network(module_module)

    return expr_mat_time, module_module

def wgcna_with_similarity_scores(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    expr_mat_time = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])
    similarity_matrices_local_and_atted(expr_mat_time, data_params['atted_path'],
                                        out_path=experiment_path)

def drought_data_e2e_pipeline(experiment_path):
    # Load the config file
    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)
    expr_mat_time = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])
    abs_dists = False
    skip_stuff = False
    if not skip_stuff:
        linkage_matrices = sum_local_distance_and_atted(
            expr_mat_time.get_distance_matrix(absolute_dist=abs_dists),
            data_params['atted_path'],
            out_path=experiment_path)

    expr_mat_time, module_module = assign_clusters_and_infer_intermodular_network(
        experiment_path=experiment_path,
        expr_mat_time=expr_mat_time,
        summed_linkage_matrix=data_params['linkage_path'],
        summed_dist_matrix_path=Path(data_params['dist_matrix_path']),
        nr_clusters=hyper_params['nr_clusters'],
        edge_cor_threshold=hyper_params['edge_corr_threshold'],
        top_nr_clusters=hyper_params['top_nr_clusters'],
        tf2_in_name=data_params['tf2_in_name'],
        tf2_out_name=data_params['tf2_out_name'])
    with (experiment_path / 'module_network.pkl').open('wb') as f:
        pickle.dump(module_module, f)
    # Assure that data has already been clustered
    assert expr_mat_time.has_been_clustered
    # expr_mat_time.get_genes_per_cluster()[328]
    my_ode = OdeModel.construct_from_regulatory_network(module_module,
                                                        nonlinear=True)

    # These are parameters that are different between the two datasets
    # They are the initial values, and the drought treatment (i.e. u_t function)
    custom_params = dict()
    small_constant = 1
    control_name = 'control'
    drought_name = 'drought'
    # custom_params[drought_name] = OdeLocalParameters(
    #      u_t=(lambda t: small_constant*(100 - t * (100 - 20) / (13 * 24))))
    #
    # custom_params[control_name] = OdeLocalParameters(
    #      u_t=(lambda t: small_constant*(90 - t * 0)))

    custom_params[control_name] = OdeLocalParameters(
        u_t=(lambda t: 0))
    custom_params[drought_name] = OdeLocalParameters(
        u_t=(lambda t: small_constant * t / (13 * 24)))

    best_ode_fit = fit_ode_to_two_datasets(
        my_ode,
        expr_mat_time,
        custom_params=custom_params,
        nr_ode_iters=hyper_params['nr_ode_iters'],
        experiment_path=experiment_path
    )
    with (experiment_path / 'pickled_ode_model.pkl').open('wb') as f:
        pickle.dump(best_ode_fit, f)

def config_preprocess(experiment_path):
    config_path = experiment_path / 'config.yaml'
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    data_params = config['data']
    hyper_params = config['hyperparams']
    experiment_params = config['experiment_data']
    agg_method_dict = {'mean': AggregationMethod.MEAN,
                       'eigengene': AggregationMethod.EIGENGENE}
    hyper_params['agg_method'] = agg_method_dict[hyper_params['agg_method']]
    return data_params, hyper_params, experiment_params


def exploratory_heat_data_scripts(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)
    explore_emtab_375(experiment_path=experiment_path,
                      in_file_path=Path(data_params['soft_path']),
                      summed_linkage_matrix=data_params['linkage_path'],
                      summed_dist_matrix_path=Path(
                          data_params['dist_matrix_path']),
                      nr_clusters=hyper_params['nr_clusters'],
                      do_log2=hyper_params['do_log2'],
                      agg_method=hyper_params['agg_method'],
                      gpl_path=data_params['gpl_path'])


def heat_data_e2e_pipeline(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    expr_mat_time = expr_mat_from_emexp(data_params['in_path'],
                                        hyper_params['agg_method'],
                                        hyper_params['do_log2'],
                                        )

    skip_stuff = True
    if not skip_stuff:
        linkage_matrices = sum_local_distance_and_atted(
            expr_mat_time.get_distance_matrix(),
            data_params['atted_path'],
            out_path=experiment_path)


    expr_mat_time, module_module =  assign_clusters_and_infer_intermodular_network(
        experiment_path=experiment_path,
        expr_mat_time=expr_mat_time,
        summed_linkage_matrix=data_params['linkage_path'],
        summed_dist_matrix_path=Path(data_params['dist_matrix_path']),
        nr_clusters=hyper_params['nr_clusters'],
        edge_cor_threshold=hyper_params['edge_corr_threshold'],
        top_nr_clusters=hyper_params['top_nr_clusters'],
        tf2_in_name=data_params['tf2_in_name'],
        tf2_out_name=data_params['tf2_out_name'])

    with (experiment_path / 'module_network.pkl').open('wb') as f:
        pickle.dump(module_module, f)
    # Assure that data has already been clustered
    assert expr_mat_time.has_been_clustered
    # expr_mat_time.get_genes_per_cluster()[328]
    my_ode = OdeModel.construct_from_regulatory_network(module_module,
                                                        nonlinear=True)

    # These are parameters that are different between the two datasets
    custom_params = dict()
    control_name =  '21'
    treatment_name = '32'

    custom_params[control_name] = OdeLocalParameters(
        u_t=(lambda t: 0))
    custom_params[treatment_name] = OdeLocalParameters(
        u_t=(lambda t: 1))

    best_ode_fit = fit_ode_to_two_datasets(
        my_ode,
        expr_mat_time,
        custom_params=custom_params,
        nr_ode_iters=hyper_params['nr_ode_iters'],
        experiment_path=experiment_path,
        param_limit=hyper_params.get('param_limit')
    )
    with (experiment_path / 'pickled_ode_model.pkl').open('wb') as f:
        pickle.dump(best_ode_fit, f)


def fit_ode_to_two_datasets(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries,
        nr_ode_iters: int,
        custom_params: Dict,
        experiment_path: Path|None = None,
        param_limit: float = .1
        ):

    # condition_names = list(custom_params.keys())
    # # Step uno
    # my_fitter = OdeFitterMultipleDatasets(my_ode,
    #                                       my_time_series_expressions,
    #                                       custom_params,
    #                                       param_limit=param_limit)
    # my_fitter.fit(max_iter=400)
    # my_fitter.calculate_current_best_fits()

    # # Make the =0 params where we think is appropriate
    # new_params = Parameters()
    # for param_name in my_fitter.master_params:
    #     if 'k_' in param_name:
    #         new_params.add(param_name, value=22)
    #     # elif 'delta' in param_name:
    #     #     new_params.add(param_name, value=0, vary=True)
    #     elif param_name in ['gamma_1', 'gamma_2']:
    #         new_params.add(param_name, value=0, vary=False)
    #     elif param_name == 'gamma_0':
    #         new_params.add(param_name, value=0.005, vary=False)
    #
    # my_fitter.master_params = new_params

    # my_fitter.fit(max_iter=500)
    # my_fitter.calculate_current_best_fits()
    # my_fitter.all_fitters[0].plot_hill_equation_range()

    multiple_fitters = [
        OdeFitterMultipleDatasets(
            my_ode, my_time_series_expressions,
            custom_params,
            param_limit=param_limit,
        ) for _ in range(5)]

    best_fit = fit_multiple_fitters(multiple_fitters, nr_ode_iters)
    best_fit.calculate_current_best_fits(data_point_overlay=True,
                                         use_err_bars=True,
                                         out_path=experiment_path / 'final_ode_fit.svg')
    return best_fit
    # multiple_fitter.fit(100)
    # best_fits = multiple_fitter.calculate_current_best_fits()
