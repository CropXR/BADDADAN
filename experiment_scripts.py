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
    explore_emtab_375, compare_clusterings_for_ode_use, pipeline_emtab_375_full
from data_wrangling import expr_mat_from_emexp, expr_mat_from_drought

from exploring_questions import plot_module_size_distributions
from helpers import get_info_from_emtab375


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
                soft_file_path=data_params['soft_path'],
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

    with mlflow.start_run(description=experiment_params['description']):
        # mlflow.log_params(data_params)
        # mlflow.log_params(hyper_params)
        mlflow.set_tags(experiment_params)
        for file in experiment_path.iterdir():
            mlflow.log_artifact(str(file))
            # if not file.suffix in ['.npy', '.pkl', '.gzip']:
            #     mlflow.log_artifact(str(file))

        # mlflow.log_artifacts(str(experiment_path))
        # mlflow.log_metrics({'bic': best_ode_fit.sol.bic,
        #                     'chi_sqr': best_ode_fit.sol.chisqr})
        # mlflow.log_image()
        # mlflow.register_model()


def module_size_pipeline(experiment_path):
    for file in experiment_path.iterdir():
        if file.name.endswith('expr_mat_dict.pkl'):
            plot_module_size_distributions(file)
    with mlflow.start_run():
        for file in experiment_path.iterdir():
            mlflow.log_artifact(str(file))
            # if not file.suffix in ['.npy', '.pkl', '.gzip']:
            #     mlflow.log_artifact(str(file))


def drought_data_e2e_pipeline(experiment_path):
    # Load the config file
    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)
    expr_mat_time = expr_mat_from_drought(data_params['soft_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])

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
                      sample_name_parser_func=get_info_from_emtab375,
                      summed_linkage_matrix=data_params['linkage_path'],
                      summed_dist_matrix_path=Path(
                                data_params['dist_matrix_path']),
                      edge_cor_threshold=None,
                      nr_clusters=hyper_params['nr_clusters'],
                      top_nr_clusters=None,
                      do_log2=hyper_params['do_log2'],
                      agg_method=hyper_params['agg_method'],
                      gpl_path=data_params['gpl_path'])


def heat_data_e2e_pipeline(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    expr_mat_time = expr_mat_from_emexp(data_params['soft_path'],
                                        hyper_params['agg_method'],
                                        hyper_params['do_log2'],
                                        data_params['gpl_path'])
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
        experiment_path=experiment_path
    )
    with (experiment_path / 'pickled_ode_model.pkl').open('wb') as f:
        pickle.dump(best_ode_fit, f)


def fit_ode_to_two_datasets(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries,
        nr_ode_iters: int,
        custom_params: Dict,
        experiment_path: Path|None = None,
        ):


    # Step uno
    # my_fitter = OdeFitterMultipleDatasets(
    #         my_ode, my_time_series_expressions, condition_names,
    #         custom_params,
    #         param_limit=.5, aggregation_method=AggregationMethod.MEAN)

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
    # Step uno
    condition_names = list(custom_params.keys())
    multiple_fitters = [
        OdeFitterMultipleDatasets(
            my_ode, my_time_series_expressions, condition_names,
            custom_params,
            param_limit=.1,
            aggregation_method=AggregationMethod.MEAN
        ) for _ in range(5)]
    best_fit = fit_multiple_fitters(multiple_fitters, nr_ode_iters)
    best_fit.calculate_current_best_fits(data_point_overlay=True,
                                         use_err_bars=True,
                                         out_path=experiment_path / 'final_ode_fit.svg')
    return best_fit
    # multiple_fitter.fit(100)
    # best_fits = multiple_fitter.calculate_current_best_fits()
