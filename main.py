import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from lmfit import fit_report
from sklearn.metrics import adjusted_rand_score

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from DynamicModels.OdeFitter import OdeFitter
from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionArrayAnnotation import ExpressionArrayAnnotation
from Expressions.ExpressionMatrix import ExpressionMatrix, \
    ExpressionMatrixTimeSeries
from helpers import plot_y_and_y_hat, fit_spline
from DynamicModels.helper_scripts_for_fitting import fit_multiple_fitters
from predict_from_static_expressions import plot_pred_vs_real

# pd.options.display.width = 0
# GEOparse.logger.set_verbosity('INFO')
logging.basicConfig(level=logging.INFO)


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
    expression_annotation = ExpressionArrayAnnotation(annotation_path)
    match expression_path.suffix:
        case '.soft':
            logging.info('Detected .soft file')
            expression_matrix = ExpressionMatrix.from_geo_file(expression_path,
                                                               expression_annotation,
                                                               log2_transform=log2_transform)
        case '.csv':
            logging.info('Detected .csv file')
            expression_matrix = ExpressionMatrix.from_csv(expression_path,
                                                          expression_annotation,
                                                          log2_transform,
                                                          csv_separator)
        case _:
            raise NotImplementedError(
                'Cannot parse file format that is currently provided')
    expression_matrix.df.to_csv(output_path)
    logging.info(f'Successfullly saved output to {output_path}')


def fit_ode_to_data(module_network: ModuleRegulatoryNetwork,
                    my_time_series_expressions: ExpressionMatrixTimeSeries):
    # Assume that data has already been clustered
    assert my_time_series_expressions.has_been_clustered
    my_time, my_data = \
        my_time_series_expressions.get_clusters_expressions_with_time(0)

    plot_y_and_y_hat(my_data, my_time)
    # my_time_series_expressions.get_genes_per_cluster()

    # module_network.plot_network()
    my_ode = OdeModel.construct_from_regulatory_network(module_network,
                                                        nonlinear=True)

    # nr_fits = 5
    # fitters = [OdeFitter(my_ode, my_data, my_time,
    #                      heat_end_time=3, param_limit=100)
    #            for _ in range(nr_fits)]
    # best_fit = fit_multiple_fitters(fitters)
    # logging.info(f'Best fit parameters {best_fit.params.pretty_print()}')
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
        "non_heat_temp": 0.25411295,
        "y0": 2.74449959,
        "y1": 2.78200775,
        "y2": 3.12812166,
        "y3": 4.94547767,
        "heat_end_time": 3,
        "heat_temp": 0.74588705 == '1 - non_heat_temp'}

    best_fit = OdeFitter(my_ode, my_data, my_time, heat_end_time=3, param_limit=100)
    for param_name in best_fit.params.valuesdict():
        if param_name != 'heat_temp':
            # Heat temp is already restrained as 1-non_heat_temp
            best_fit.params[param_name].set(value=best_params_so_far[param_name])
    best_fit.params["heat_end_time"].set(value=3, vary=False)
    best_fit.params["heat_temp"].set(expr='1 - non_heat_temp')
    best_fit.fit()

    # more_time = np.linspace(0, 24, 50)
    # Next step: simulate the data with these params
    simulated_data = best_fit.calculate_current_best_fit(my_time)

    plot_y_and_y_hat(y_real=my_data, t_real=my_time,
                     model_fit=simulated_data)
    #
    # # And try to best_fit model again
    # try_again = OdeFitter(my_ode, simulated_data.y, simulated_data.t)
    # second_fit = try_again.best_fit()
    # fit_to_simul = try_again.predict_values(second_fit.params, simulated_data.t)
    #
    # plot_y_and_y_hat(y_real=simulated_data.y, t_real=simulated_data.t,
    #                  model_fit=fit_to_simul)
    # logging.info(fit_report(second_fit))
    # # print(fit_report(second_fit))


def thickening_thinning(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries,
        std_cutoff=1.5
):
    n_clusters = 4
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    my_time_series_expressions.keep_genes_above_deviation_cutoff(
        cutoff=std_cutoff)
    my_time, my_data = \
        my_time_series_expressions.get_clusters_expressions_with_time(
            n_clusters)
    my_time_series_expressions.get_genes_per_cluster()
    # Interpolate data
    interp_time, interp_data = fit_spline(my_data, my_time, num_timepoints=15)
    my_time, my_data = interp_time, interp_data
    initial_sim_fit = OdeFitter(my_ode, my_data, my_time)
    # Note: look into the initial parameter values
    # optimal_fit = initial_sim_fit.fit(method='differential_evolution')
    optimal_fit = initial_sim_fit.thickening_thinning(nr_rounds=3)

    predicted_values = initial_sim_fit.calculate_current_best_fit(my_time)

    plot_y_and_y_hat(y_real=my_data, t_real=my_time,
                     model_fit=predicted_values)

    logging.info(fit_report(optimal_fit))

def compare_clusterings(
        cluster_path1: Path,
        cluster_path2: Path,
):
    """Compare the clusterings created by two different
    pipelines to see if they agree. E.g. you can pass it two
    gene_to_module.csv files to compare them
    """
    df1 = pd.read_csv(cluster_path1, sep=' ', names=['gene', 'module'])
    df2 = pd.read_csv(cluster_path2, sep=' ', names=['gene', 'module'])

    merged_df = df1.merge(df2, on='gene')
    agreement_score = adjusted_rand_score(merged_df.module_x.to_list(), merged_df.module_y.to_list())
    print(agreement_score)

if __name__ == "__main__":
    typer.run(annotate_microarray_expression)
