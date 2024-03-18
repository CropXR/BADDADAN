import copy
import logging
from pathlib import Path
import dill as pickle

import numpy as np
import pandas as pd
import typer
import yaml
from lmfit import create_params, Parameters
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score
import mlflow

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from DynamicModels.OdeFitter import OdeFitter
from DynamicModels.OdeFitterMultipleDatasets import OdeFitterMultipleDatasets
from DynamicModels.OdeLocalParameters import OdeLocalParameters
from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import ExpressionMatrix, \
    ExpressionMatrixTimeSeries, AggregationMethod
from analysis_pipelines import pipeline_from_atted_clustering, \
    local_clustering_on_atted_clusters, pipeline_from_summed_clustering, \
    compare_clusterings_for_ode_use
from exploring_questions import save_local_distance_matrix, \
    sum_local_distance_and_atted, rand_index_both_clusterings
from helpers import plot_y_and_y_hat, get_info_from_gse65046
from DynamicModels.helper_scripts_for_fitting import fit_multiple_fitters

# pd.options.display.width = 0
# GEOparse.logger.set_verbosity('INFO')

# logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

def annotate_microarray_expression(
        expression_path: Path = typer.Option(
            ..., help='Path to geo expression file. Works on .soft and .csv '
                      'format. Others have not been tested.'),
        output_path: Path = typer.Option(
            ..., help='Path to filename where .csv of annotated expression'
                      ' data will be saved.'),
        annotation_path: Path = typer.Option(
            Path(
                f'{__file__}/../data/resources/affy_ATH1_array_elements-2010-12-20.txt').resolve(),
            help='Path to annotation of micro array. '
                 'Arabidopsis ATH1 annotation is provided by default.'),
        log2_transform: bool = typer.Option(
            False, help='Perform log2 transformation on expression data.'),
        csv_separator: str = typer.Option(
            '\t', help='Seperator to use when splitting csv columns')
):
    """From a geo expression file and annotation file. Generate output file
    where microarray expressions use gene labels (e.g. AT1G65110) instead of
    the default affymetrix probe_ids (e.g. 263139_at)
    """
    match expression_path.suffix:
        case '.soft':
            logging.info('Detected .soft file')
            expression_matrix = ExpressionMatrix.from_geo_file(expression_path,
                                                               log2_transform=log2_transform)
        case '.csv':
            logging.info('Detected .csv file')
            expression_matrix = ExpressionMatrix.from_csv(expression_path,
                                                          log2_transform,
                                                          csv_separator)
        case _:
            raise NotImplementedError(
                'Cannot parse file format that is currently provided')
    expression_matrix.df.to_csv(output_path)
    logging.info(f'Successfullly saved output to {output_path}')


def full_pipeline_prototype(out_dir: Path,
                            input_expression_file: Path,
                            nr_genes: int = 400,
                            nr_clusters: int = 4,
                            do_log2: bool = True,
                            ):
    """Get raw data, apply clustering from Jordi, and try to fit ODE model from that
    """
    # Annotate genes, log2 transform them
    out_dir.mkdir(parents=True, exist_ok=True)

    # expr_mat_time = ExpressionMatrixTimeSeries.from_csv(input_expression_file,
    #                                                     log2_transform=do_log2)

    expr_mat_time = ExpressionMatrixTimeSeries.from_geo_file(input_expression_file,
                                                        log2_transform=do_log2,
                                                             annotate_from_gpl=True)
    # TODO filter out low expression genes

    expr_mat_time.keep_n_most_deviating_genes(nr_genes)
    expr_mat_time.plot_sample_gene_heatmap()
    expr_mat_time.do_hierachical_clustering(nr_clusters, do_plotting=True)
    expr_mat_time.column_parser = get_info_from_gse65046

    phenotype_dict = {
        'drought': {'stomatal_conductance':
                        [150, 240, 240, 120, 100, 100, 110, 80, 110, 120, 100, 50, 50, 20]
                    },
        'control': {'stomatal_conductance':
                        [160, 250, 210, 120, 120, 140, 140, 150, 180, 230, 200, 110, 160, 170]
                    }
    }

    expr_mat_drought = copy.deepcopy(expr_mat_time)
    expr_mat_drought.keep_only_samples_with_string('drought')
    expr_mat_drought.plot_clusters_over_time(title='Drought')
    # expr_mat_drought.plot_clusters_over_time(title='Drought', plot_units=True)

    expr_mat_control = copy.deepcopy(expr_mat_time)
    expr_mat_control.keep_only_samples_with_string('control')
    expr_mat_control.plot_clusters_over_time(title='Control')
    # expr_mat_control.plot_clusters_over_time(title='Control', plot_units=True)

    expr_mat_time.write_tf2_input_file(out_dir / 'tf2input.txt', omit_unannotated_genes=True)

    my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(
        out_dir / '02_tf2network_output.tsv')
    my_grn.add_tf_module_mappings(out_dir / 'tf2input.txt', from_tf2_input=True)

    my_grn.clean_up_network()
    my_grn.check_if_tfs_created_by_module(expr_mat_time, do_plotting=True,
                                          remove_low_corr=True)
    my_grn.set_up_or_downregulation(expr_mat_time, do_plotting=True)
    # my_grn.plot_network(nx.draw_kamada_kawai, with_labels=False)
    module_module = my_grn.get_module_module_network()
    # # module_module.graph = nx.create_empty_copy(module_module.graph, with_data=False)
    module_module.plot_network(with_labels=True)

    # expr_mat_time.assign_clusters_from_jordi_input(input_file_jordi, drop_duplicates=True)
    # expr_mat_subset.add_phenotypes(phenotype_dict[condition])
    # expr_mat_subset.corr_to_phenotypes()
    # expr_mat_subset.plot_clusters_over_time(title=condition)
    # fit_ode_to_data(module_module, expr_mat_drought)
    # fig_path = out_path / 'fitted_model.svg'
    # fit_ode_to_two_datasets(module_module, expr_mat_drought, expr_mat_control, fig_path)
    #

def fit_ode_to_two_simulated_data(module_network: ModuleRegulatoryNetwork):
    """Trying this with parameters from the 500 highest MAD log2 genes,
    with own connection added in the 03_tf2network_output.
    """
    my_ode = OdeModel.construct_from_regulatory_network(module_network,
                                                        nonlinear=True)

    logging.info(my_ode)
    param_dict = {'delta_0': 0.12751760148586033,
                  'gamma_0': -0.9952741725598351,
                  'beta_1_0': 23.022511281793502,
                  'k_1_0': 79.65334412365583,
                  'delta_1': 0.06735892650098774,
                  'gamma_1': 0.37971918057742826,
                  'beta_3_1': 48.84538036408849,
                  'k_3_1': 4.8659037910070424e-08,
                  'beta_2_1': 0.0002427304981234002,
                  'k_2_1': 0.5477925420514451,
                  'delta_2': 0.043497290440686065,
                  'gamma_2': 0.0859898629842899,
                  'delta_3': 31.483347391315707,
                  'gamma_3': 6.833788089421034,
                  'beta_1_3': 98.54535183271824,
                  'k_1_3': 41.39249687732453,
                  'beta_2_3': 1.3836725778526238,
                  'k_2_3': 85.43469800417957,
                  'heat_temp': 0.9,
                  'non_heat_temp': 0.1,
                  'heat_end_time': 3,
                  'y0': 3.130649824906166,
                  'y1': 2.238649944776142,
                  'y2': 5.29793208572219,
                  'y3': 4.060684898675729
                  }
    my_params = Parameters()
    for key, value in param_dict.items():
        my_params.add(key, value)
    my_time = np.array([0.25, 0.5, 1., 3., 4., 6., 12., 24.])
    sim_exp_data = my_ode.calculate_solution(my_params,
                                             my_time,
                                             init_condition_names=[f'y{i}' for i
                                                                   in range(4)]
                                             )
    sim_exp_matrix = ExpressionMatrixTimeSeries.from_simulated_data(
        sim_exp_data)
    plot_y_and_y_hat(sim_exp_data.y, my_time)
    plt.show()
    my_params['heat_end_time'].set(value=-1)
    sim_control_data = my_ode.calculate_solution(my_params,
                                                 my_time,
                                                 init_condition_names=[f'y{i}'
                                                                       for i in
                                                                       range(4)]
                                                 )
    sim_control_matrix = ExpressionMatrixTimeSeries.from_simulated_data(
        sim_control_data)
    plot_y_and_y_hat(sim_control_data.y, my_time)
    plt.show()

    # These are parameters that are different between the two fitters,
    # some are fixed (e.g. the heat_end_time) and some are changed
    # during training (e.g. y0, non_heat_temp)

    custom_params = {sim_exp_matrix: create_params(heat_end_time=3.,
                                                   y0=1,
                                                   y1=1,
                                                   y2=1,
                                                   y3=1,
                                                   non_heat_temp=.1,
                                                   heat_temp=.9),
                     sim_control_matrix: create_params(heat_end_time=-1.,
                                                       y0=1,
                                                       y1=1,
                                                       y2=1,
                                                       y3=1,
                                                       non_heat_temp=.1,
                                                       heat_temp=.9),
                     }

    # Create the fitter here, which contains this collection of custom parameters
    # multiple_fitter = OdeFitterMultipleDatasets(
    #     my_ode, [sim_control_matrix, sim_exp_matrix], custom_params,
    #     param_limit=150)

    # # Slightly perturb initial parameters
    # for param_name, value in my_params.valuesdict().items():
    #     my_params[param_name].set(value=np.random.normal(value, 0.5 * abs(value)))
    # # # Provide with prior knowledge on ground truth parameters
    # multiple_fitter.master_params = my_params

    # multiple_fitter.calculate_current_best_fits()
    my_ode.flip_regulatory_sign(1, 3)
    nr_fits = 4
    fitters = [OdeFitterMultipleDatasets(
                my_ode, [sim_control_matrix, sim_exp_matrix],
                custom_params, param_limit=150)
               for _ in range(nr_fits)]

    for fitter in fitters:
        new_params = Parameters()
        # Slightly perturb initial parameters
        for param_name, value in my_params.valuesdict().items():
            if param_name in fitter.master_params:
                new_params.add(name=param_name,
                               value=np.random.normal(value, 0.1 * abs(value)))
        # # Provide with prior knowledge on ground truth parameters
        fitter.master_params = my_params

    best_fit = fit_multiple_fitters(fitters, nr_iters=1000) #, extra_analysis=True,gt_params=my_params)
    best_fit.master_params.pretty_print()
    best_fit.calculate_current_best_fits()
    logging.info(f'Best fit parameters {best_fit.master_params.valuesdict()}')

    # multiple_fitter.fit(50)
    # multiple_fitter.calculate_current_best_fits()


def fit_ode_to_two_datasets(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries,
        nr_ode_iters: int,
        experiment_path: Path|None = None,
        ):

    # These are parameters that are different between the two datasets
    # They are the initial values, and the drought treatment (i.e. u_t function)
    custom_params = dict()
    small_constant = 1
    control_name = 'control'
    drought_name = 'drought'
    condition_names = [control_name, drought_name]
    # custom_params[drought_name] = OdeLocalParameters(
    #      u_t=(lambda t: small_constant*(100 - t * (100 - 20) / (13 * 24))))
    #
    # custom_params[control_name] = OdeLocalParameters(
    #      u_t=(lambda t: small_constant*(90 - t * 0)))


    custom_params[control_name] = OdeLocalParameters(
         u_t=(lambda t: 0))
    custom_params[drought_name] = OdeLocalParameters(
         u_t=(lambda t: small_constant * t / (13*24)))

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


def fit_ode_to_data(module_network: ModuleRegulatoryNetwork,
                    my_time_series_expressions: ExpressionMatrixTimeSeries):
    # Assume that data has already been clustered
    assert my_time_series_expressions.has_been_clustered
    # my_time_series_expressions.plot_clusters_over_time(plot_units=True)
    my_time, my_data = my_time_series_expressions.get_clusters_expressions_with_time(
        0)

    plot_y_and_y_hat(my_data, my_time)
    # my_time_series_expressions.get_genes_per_cluster()

    # module_network.plot_network()
    my_ode = OdeModel.construct_from_regulatory_network(module_network,
                                                        nonlinear=True)
    logging.info(my_ode)

    # nr_fits = 5
    # fitters = [OdeFitter(my_ode, my_data, my_time,
    #                      heat_end_time=3, param_limit=100)
    #            for _ in range(nr_fits)]
    # best_fit = fit_multiple_fitters(fitters, nr_iters=1000, extra_analysis=False)
    # best_fit.params.pretty_print()
    # logging.info(f'Best fit parameters {best_fit.params.valuesdict()}')

    best_params_so_far = {
        "delta_0": 6.96879050,
        "gamma_0": 4.52378448,
        "beta_2_0": 6.85200720,
        "k_2_0": 72.1041216,
        "beta_3_0": 9.88057244,
        "k_3_0": 56.8076177,
        "delta_1": 0.40712237,
        "gamma_1": -0.14602615,
        "beta_0_1": 91.5736595,
        "k_0_1": 5.1010e-05,
        "beta_2_1": 2.35521423,
        "k_2_1": 95.7456557,
        "beta_3_1": 74.0149223,
        "k_3_1": 0.09908515,
        "delta_2": 0.03132386,
        "gamma_2": -2.76555243,
        "beta_3_2": 11.3435667,
        "k_3_2": 0.85995851,
        "beta_0_2": 0.72054543,
        "k_0_2": 76.6528692,
        "delta_3": 0.01680434,
        "gamma_3": -0.49699132,
        "beta_1_3": 6.3335e-04,
        "k_1_3": 85.5113151,
        "beta_0_3": 99.2512088,
        "k_0_3": 0.02191174,
        "y0": 2.74449959,
        "y1": 2.78200775,
        "y2": 3.12812166,
        "y3": 4.94547767,
        "non_heat_temp": 0.25411295,
    }

    best_fit = OdeFitter(my_ode, my_data, my_time, param_limit=100)
    for param_name in best_fit.params.valuesdict():
        if param_name not in ['heat_temp', 'heat_end_time']:
            # Heat temp is already restrained as 1-non_heat_temp
            best_fit.params[param_name].set(
                value=best_params_so_far[param_name])
    best_fit.params["heat_end_time"].set(value=-1, vary=False)
    best_fit.params["heat_temp"].set(expr='1 - non_heat_temp')
    best_fit.fit(100)

    # more_time = np.linspace(0, 24, 50)

    # Next step: simulate the data with these params
    simulated_data = best_fit.calculate_current_best_fit(my_time)
    # best_fit.plot_hill_equation_range(my_time)

    plot_y_and_y_hat(y_real=my_data, t_real=my_time,
                     model_fit=simulated_data)
    plt.show()

    # # And try to best_fit model again
    # try_again = OdeFitter(my_ode, simulated_data.y, simulated_data.t)
    # second_fit = try_again.best_fit()
    # fit_to_simul = try_again.predict_values(second_fit.params, simulated_data.t)
    #
    # plot_y_and_y_hat(y_real=simulated_data.y, t_real=simulated_data.t,
    #                  model_fit=fit_to_simul)
    # logging.info(fit_report(second_fit))
    # # print(fit_report(second_fit))


def compare_clusterings(
        cluster_path1: Path,
        cluster_path2: Path,
):
    """Compare the clusterings created by two different
    pipelines to see if they agree. E.g. you can pass it two
    gene_to_module.csv files to compare them
    """
    df1 = pd.read_csv(cluster_path1, sep=' ', names=['module', 'gene'])
    df2 = pd.read_csv(cluster_path2, sep=' ', names=['module', 'gene'])

    merged_df = df1.merge(df2, on='gene')
    agreement_score = adjusted_rand_score(merged_df.module_x.to_list(),
                                          merged_df.module_y.to_list())
    print()
    logging.info(f'Nr of overlapping genes {len(merged_df)}')
    logging.info(agreement_score)
    print()


def count_flowering_genes(path_to_gene_selection: Path,
                          path_to_flowering_genes_pkl: Path):
    selected_gene_df = pd.read_csv(path_to_gene_selection, sep=' ',
                                   names=['module', 'gene'])
    flowering_gene_df = pd.read_pickle(path_to_flowering_genes_pkl)
    overlap_df = selected_gene_df.merge(flowering_gene_df, left_on='gene',
                                        right_on='locustag')
    return overlap_df


def camila_red_panda(soft_file_in_path: Path,
                     out_dir: Path,
                     do_log2: bool = True,
                     variation_measure: str = 'mad',
                     nr_total_genes: int = 2000,
                     nr_clusters: int = 4):
    """Minimum working example on how to do the full procedure of
    parsing expression data, clustering it, and fitting model to it.

    Currently I'm exposing quite a bit of code to you, but I think that
    will help you fix things if you ever run into issues ;)
    """
    expression_matrix = ExpressionMatrixTimeSeries.from_geo_file(
        soft_file_in_path, log2_transform=do_log2)

    # # Uncomment if you want to use a second expressionmatrix
    # # during clustering as well
    # control_expr_mat_time = ExpressionMatrixTimeSeries.from_geo_file(
    #     some_other_in_path, log2_transform=do_log2)
    # expression_matrix.concat_to_expression_matrix(control_expr_mat_time,
    #                                           keys=['Heat', 'Control'])

    expression_matrix.keep_n_most_deviating_genes(nr_total_genes,
                                                  variation_measure)
    expression_matrix.do_hierachical_clustering(n_cluster=nr_clusters,
                                                do_plotting=False)
    # # Also uncomment if you want to use a second expressionmatrix
    # # during clustering
    # expression_matrix.remove_condition_from_expression_matrix('Control')
    # control_expr_mat_time.assign_clusters_from(expr_mat_time)

    # Create a file that can be used on http://bioinformatics.psb.ugent.be/webtools/TF2Network/
    # to get putative regulators per cluster
    expression_matrix.write_tf2_input_file(
        out_dir / f'01_tf2network_input_{nr_total_genes}_highest_{variation_measure}_genes.txt',
        omit_unannotated_genes=True)
    # On TF2Network, make sure to select direct_export
    expression_matrix.save_tf_produced_by_module_file(
        out_dir / f'02_gene_to_module.csv',
        tf_list_path=Path('data/resources/Ath_TF_list.txt')
    )

    assert (out_dir / '03_tf2network_output.tsv').exists(), 'Save TF2Network output first'
    # From TF2 output, create gene regulatory network
    my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(
        out_dir / '03_tf2network_output.tsv')
    my_grn.add_tf_module_mappings(out_dir / '02_gene_to_module.csv')
    my_grn.clean_up_network()
    # my_grn.plot_network(with_labels=True)
    my_grn.check_if_tfs_created_by_module(expression_matrix, do_plotting=False,
                                          remove_low_corr=True)
    my_grn.set_up_or_downregulation(expression_matrix, do_plotting=False)
    module_module = my_grn.get_module_module_network()
    module_module.plot_network(with_labels=True)

    # Assume that data has already been clustered
    assert expression_matrix.has_been_clustered
    my_time, my_data = expression_matrix.get_clusters_expressions_with_time(
        0, aggregation_method='mean')
    # my_time_series_expressions.plot_clusters_over_time(plot_units=True)

    # Create system of ordinary differential equations
    my_ode = OdeModel.construct_from_regulatory_network(module_module,
                                                        nonlinear=True)
    logging.info(my_ode)

    # Fit using gradient descent with multiple starting
    nr_fits = 5
    fitters = [OdeFitter(my_ode, my_data, my_time, param_limit=100)
               for _ in range(nr_fits)]
    best_fit = fit_multiple_fitters(fitters, nr_iters=1000, extra_analysis=False)
    best_fit.params.pretty_print()
    logging.info(f'Best fit parameters {best_fit.params.valuesdict()}')

    # See how best fit looks compared to the experimental data
    simulated_data = best_fit.calculate_current_best_fit(my_time)
    # best_fit.plot_hill_equation_range(my_time)

    plot_y_and_y_hat(y_real=my_data, t_real=my_time,
                     model_fit=simulated_data)
    plt.show()

def main():
    # ONLY EDIT THESE LINES
    experiment_path = Path('data/experiments/04_comparing_clusterings')
    mlflow.set_experiment("/04_comparing_clusterings")

    ##  This all shouldn't have to be changed ##
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(experiment_path / "log.log"),
                                  logging.StreamHandler()])
    # Load the config file
    config_path = experiment_path / 'config.yaml'
    with config_path.open('r') as f:
        config = yaml.safe_load(f)

    data_params = config['data']
    hyper_params = config['hyperparams']

    agg_method_dict = {'mean': AggregationMethod.MEAN,
                       'eigengene': AggregationMethod.EIGENGENE}
    hyper_params['agg_method'] = agg_method_dict[hyper_params['agg_method']]

    compare_clusterings_for_ode_use(
        experiment_path=experiment_path,
        soft_file_path=Path(data_params['soft_path']),
        agg_method=hyper_params['agg_method'], do_log2=hyper_params['do_log2'],
        summed_linkage_matrix=data_params['linkage_path'],
        summed_dist_matrix_path=Path(data_params['dist_matrix_path']),
        atted_linkage_matrix=Path(data_params['atted_linkage_matrix']),
        atted_path=Path(data_params['atted_path']),
        nr_clusters=hyper_params['nr_clusters'],
        edge_cor_threshold=hyper_params['edge_corr_threshold'],
        top_nr_clusters=hyper_params['top_nr_clusters'],
        tf2_in_name=data_params['tf2_in_name'],
        tf2_out_name=data_params['tf2_out_name'],
    )

    with mlflow.start_run(
            description=config['experiment_data']['description']):
        mlflow.log_params(data_params)
        mlflow.log_params(hyper_params)
        mlflow.set_tags(config['experiment_data'])
        mlflow.log_artifacts(str(experiment_path))
        # mlflow.log_metrics({'bic': best_ode_fit.sol.bic,
        #                     'chi_sqr': best_ode_fit.sol.chisqr})
        # mlflow.log_image()
        # mlflow.register_model()

    # expr_mat_time, module_module, atted_stats = pipeline_from_atted_clustering(
    #     experiment_path=experiment_path,
    #     soft_file_path=Path(data_params['soft_path']),
    #     atted_linkage_matrix=data_params['atted_linkage_matrix'],
    #     atted_path=data_params['atted_path'],
    #     edge_cor_threshold=hyper_params['edge_corr_threshold'],
    #     nr_clusters=hyper_params['nr_clusters'],
    #     top_nr_clusters=hyper_params['top_nr_clusters'],
    #     do_log2=hyper_params['do_log2'],
    #     agg_method=hyper_params['agg_method'])

    # pipeline_only_local_clustering()



    # Assure that data has already been clustered
    assert expr_mat_time.has_been_clustered
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



if __name__ == "__main__":
    main()
