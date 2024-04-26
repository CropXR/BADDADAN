from pathlib import Path

import dill as pickle
import mlflow
import yaml

from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import AggregationMethod
from analysis_pipelines import pipeline_from_summed_clustering, \
    pipeline_emtab_375_full

from exploring_questions import plot_module_size_distributions
from helpers import get_info_from_emtab375
from main import fit_ode_to_two_datasets


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
    data_params, hyper_params = config_preprocess(experiment_path)

    expr_mat_time, module_module = pipeline_from_summed_clustering(
        experiment_path=experiment_path,
        soft_file_path=data_params['soft_path'],
        agg_method=hyper_params['agg_method'],
        do_log2=hyper_params['do_log2'],
        summed_linkage_matrix=data_params['linkage_path'],
        summed_dist_matrix_path=Path(data_params['dist_matrix_path']),
        nr_clusters=hyper_params['nr_clusters'],
        edge_cor_threshold=hyper_params['edge_corr_threshold'],
        top_nr_clusters=hyper_params['top_nr_clusters'],
        tf2_in_name=data_params['tf2_in_name'],
        tf2_out_name=data_params['tf2_out_name']
    )
    with (experiment_path / 'module_network.pkl').open('wb') as f:
        pickle.dump(module_module, f)
    # Assure that data has already been clustered
    assert expr_mat_time.has_been_clustered
    expr_mat_time.get_genes_per_cluster()[328]
    my_ode = OdeModel.construct_from_regulatory_network(module_module,
                                                        nonlinear=True)
    best_ode_fit = fit_ode_to_two_datasets(
        my_ode,
        expr_mat_time,
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
    agg_method_dict = {'mean': AggregationMethod.MEAN,
                       'eigengene': AggregationMethod.EIGENGENE}
    hyper_params['agg_method'] = agg_method_dict[hyper_params['agg_method']]
    return data_params, hyper_params


def heat_data_pipeline_setup(experiment_path):
    data_params, hyper_params = config_preprocess(experiment_path)
    pipeline_emtab_375_full(experiment_path=experiment_path,
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
