import logging
from pathlib import Path

import typer
from lmfit import fit_report

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


def fit_ode_to_data(my_ode: OdeModel,
                    my_time_series_expressions: ExpressionMatrixTimeSeries):
    # Assume that data has already been clustered
    my_time, my_data = \
        my_time_series_expressions.get_clusters_expressions_with_time(0)

    best_params_so_far = {
        "delta_0": 3.0970e-04,
        "gamma_0": -3.30882487,
        "beta_1_0": 308.168437,
        "k_1_0": 1423.49934,
        "beta_2_0": 1609.77151,
        "k_2_0": 1747.61527,
        "delta_1": 0.13491609,
        "gamma_1": -0.72451590,
        "beta_2_1": 3868.00813,
        "k_2_1": 3405.72210,
        "beta_0_1": 4.79600616,
        "k_0_1": 2483.00129,
        "delta_2": 0.35237808,
        "gamma_2": 0.49643743,
        "beta_0_2": 1976.20308,
        "k_0_2": 1.2062e-04,
        "beta_1_2": 1859.86147,
        "k_1_2": 15.4168390,
        "delta_3": 5.65706872,
        "gamma_3": 3.01388575,
        "beta_2_3": 2443.55505,
        "k_2_3": 838.128554,
        "beta_1_3": 2064.37510,
        "k_1_3": 544.943017,
        "beta_0_3": 1070.24154,
        "k_0_3": 474.161978,
        "non_heat_temp": 0.71839443,
        "heat_end_time": 3,
        "y0": 421.740813,
        "y1": 1551.40704,
        "y2": 1046.54923,
        "y3": 493.981306,
    }
    # plot_y_and_y_hat(my_data, my_time)
    # my_time_series_expressions.get_genes_per_cluster()

    # Fit single model
    fit = OdeFitter(my_ode, my_data, my_time, heat_end_time=3, param_limit=4000)
    for param_name in fit.params.valuesdict():
        if param_name != 'heat_temp':
            # Heat temp is already restrained as 1-non_heat_temp
            fit.params[param_name].set(value=best_params_so_far[param_name])

    fit.fit()

    # # Fit multiple models simultaneously
    # nr_fits = 5
    # fitters = [OdeFitter(my_ode, my_data, my_time,
    #                      heat_end_time=3, param_limit=4000)
    #            for _ in range(nr_fits)]
    # fit = fit_multiple_fitters(fitters)

    # more_time = np.linspace(0, 24, 50)
    # Next step: simulate the data with these params
    simulated_data = fit.calculate_current_best_fit(my_time)

    plot_y_and_y_hat(y_real=my_data, t_real=my_time,
                     model_fit=simulated_data)
    #
    # # And try to fit model again
    # try_again = OdeFitter(my_ode, simulated_data.y, simulated_data.t)
    # second_fit = try_again.fit()
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


if __name__ == "__main__":
    typer.run(annotate_microarray_expression)
