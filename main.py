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
from predict_from_static_expressions import plot_pred_vs_real

# pd.options.display.width = 0
# GEOparse.logger.set_verbosity('INFO')
# logging.basicConfig(level=logging.INFO)


def stub_main(
        expression_path: Path = typer.Option(...,
                                             help='Path to geo expression file'),
        annotation_path: Path = typer.Option(...,
                                             help='Path to annotation of micro array'),
        n_cluster: int = typer.Option(5,
                                      help='Number of gene clusters to '
                                           'extract'),
):
    """Deprecated, might reuse this at some point though"""
    expression_annotation = ExpressionArrayAnnotation(annotation_path)
    expression_matrix = ExpressionMatrix.from_geo_file(expression_path,
                                                       expression_annotation)
    expression_matrix = expression_matrix.keep_only_wt_samples()
    plot_pred_vs_real(expression_matrix, n_cluster)
    # do_cv_for_nclust(expression_matrix)


def fit_ode_to_data(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries,
        std_cutoff=1.5
):
    n_clusters = 4
    my_time_series_expressions.keep_only_shoot()
    my_time_series_expressions.merge_biological_samples()
    my_time_series_expressions.keep_only_de_genes(std_cutoff=std_cutoff)
    my_time, my_data = \
        my_time_series_expressions.get_clusters_expressions_with_time(n_clusters)
    my_time_series_expressions.get_genes_per_cluster()
    # Interpolate data
    interp_time, interp_data = fit_spline(my_data, my_time, num_timepoints=50)
    my_time, my_data = interp_time, interp_data
    initial_sim_fit = OdeFitter(my_ode, my_data, my_time)
    # Note: look into the initial parameter values
    # optimal_fit = initial_sim_fit.fit(method='differential_evolution')
    # optimal_fit = initial_sim_fit.thickening_thinning(3)
    # optimal_fit = initial_sim_fit.fit(method='bfgs')
    optimal_fit = initial_sim_fit.fit()
    logging.info(fit_report(optimal_fit))
    # print(fit_report(optimal_fit))

    # more_time = np.linspace(0, 24, 50)
    # Next step: simulate the data with these params
    simulated_data = initial_sim_fit.predict_values(optimal_fit.params,
                                                    my_time)

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
    my_time_series_expressions.keep_only_de_genes(std_cutoff=std_cutoff)
    my_time, my_data = \
        my_time_series_expressions.get_clusters_expressions_with_time(n_clusters)
    my_time_series_expressions.get_genes_per_cluster()
    # Interpolate data
    interp_time, interp_data = fit_spline(my_data, my_time, num_timepoints=15)
    my_time, my_data = interp_time, interp_data
    initial_sim_fit = OdeFitter(my_ode, my_data, my_time)
    # Note: look into the initial parameter values
    # optimal_fit = initial_sim_fit.fit(method='differential_evolution')
    optimal_fit = initial_sim_fit.thickening_thinning(nr_rounds=3)

    predicted_values = initial_sim_fit.predict_values(optimal_fit.params, my_time)

    plot_y_and_y_hat(y_real=my_data, t_real=my_time,
                     model_fit=predicted_values)

    logging.info(fit_report(optimal_fit))


if __name__ == "__main__":
    typer.run(stub_main)
