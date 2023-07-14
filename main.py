import logging
import pickle
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import typer
from lmfit import fit_report, create_params, Parameters
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from DynamicModels.OdeFitter import OdeFitter
from DynamicModels.OdeFitterMultipleDatasets import OdeFitterMultipleDatasets
from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import ExpressionMatrix, \
    ExpressionMatrixTimeSeries
from helpers import plot_y_and_y_hat, fit_spline, get_info_from_emexp1304
from DynamicModels.helper_scripts_for_fitting import fit_multiple_fitters

# pd.options.display.width = 0
# GEOparse.logger.set_verbosity('INFO')
logging.basicConfig(level=logging.INFO)


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
    expression_matrix = ExpressionMatrixTimeSeries.from_csv(
        soft_file_in_path, log2_transform=do_log2, sep=',',
        column_decode_function=get_info_from_emexp1304)

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
        out_dir / '03_tf2network_output.tsv', nr_top_hits=10)
    my_grn.add_tf_module_mappings(out_dir / '02_gene_to_module.csv')
    my_grn.clean_up_network()
    my_grn.plot_network(with_labels=True)
    my_grn.check_if_tfs_created_by_module(expression_matrix, do_plotting=True,
                                          remove_low_corr=True)
    my_grn.set_up_or_downregulation(expression_matrix, do_plotting=True)
    module_module = my_grn.get_module_module_network()
    module_module.plot_network(with_labels=True)

    # Assume that data has already been clustered
    assert expression_matrix.has_been_clustered
    my_time, my_data = expression_matrix.get_clusters_expressions_with_time(
        0, aggregation_method='mean')
    # plot_y_and_y_hat(y_real=my_data, t_real=my_time)
    # plt.show()
    # Create system of ordinary differential equations
    my_ode = OdeModel.construct_from_regulatory_network(module_module,
                                                        nonlinear=True)
    logging.info(my_ode)

    # Fit using gradient descent with multiple starting
    nr_fits = 5
    fitters = [OdeFitter(my_ode, my_data, my_time,
                         heat_end_time=-1, param_limit=10)
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


if __name__ == "__main__":
    camila_red_panda()
