import copy
from copy import deepcopy
import logging
from itertools import combinations
from pathlib import Path
import re

import dill as pickle
import mlflow
import networkx as nx
import numpy as np
import pandas as pd
import pypesto
import pypesto.petab
import petab
import amici.petab.simulator
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.spatial.distance import squareform
import amici
from tqdm import tqdm
from statannotations.Annotator import Annotator


from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from GoEnrich.EnrichedGeneModuleGoTerms import EnrichedGeneModuleGoTerms
from data_wrangling import module_network_from_tf2_output
from expr_mat_factories import expr_mat_from_heat, expr_mat_from_drought
from exceptions import RegulatoryDisagreementError
from helpers import config_preprocess

from petab_integration.petab_scripts import write_petab_files, \
    param_optimise_petab_problem


def save_supp_table_go_enrichments(expr_mat_pickl_path, go_enrich_output_path,
                                   treatment_path):
    with expr_mat_pickl_path.open('rb') as f:
        expr_mat_time: ExpressionMatrixTimeSeries = pickle.load(f)
    expr_mat_time.save_go_enrich_supp_table(
        go_enrich_output_path,
        out_path=treatment_path / 'go_terms_supp_table.csv'
    )


def drought_from_wgcna(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    wgcna_module_assignment = data_params['wgcna_module_assignment_path']
    expr_mat_time: ExpressionMatrixTimeSeries = expr_mat_from_drought(
        data_params['in_path'], hyper_params['agg_method'],
        hyper_params['do_log2'])
    expr_mat_time.assign_clusters_from_wgcna(wgcna_module_assignment)
    expr_mat_time.plot_cluster_sizes()

    tf2_in_path =experiment_path / data_params['tf2_in_name']
    tf2_out_path = experiment_path / data_params['tf2_out_name']
    # Post to tf2network
    expr_mat_time.write_tf2_input_file(
        out_path=tf2_in_path)

    # expr_mat_time.do_genewise_normalisation()
    expr_mat_time.keep_highest_z_clusters(
        hyper_params['top_nr_clusters'],
        tf2_output_path=tf2_out_path,
    plotting_path=experiment_path)

    expr_mat_time.plot_clusters_over_time()
    # # TO get gene list
    # [print(i) for i in expr_mat_time.get_genes_per_cluster()[32]]

    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=hyper_params['edge_corr_threshold'],
        module_plot_path=experiment_path / 'global_cluster_module_network.svg')

    expr_mat_time.keep_only_modules_in_network(module_module)

    return expr_mat_time, module_module

def save_files_for_wgcna_cutting(experiment_path: Path,
                                 data_params: dict,
                                 expr_mat_time: ExpressionMatrixTimeSeries):
    for folder_name in ['figs', 'full_datasets']:
        new_folder = experiment_path / folder_name
        new_folder.mkdir(parents=True, exist_ok=True)

    fig_folder = experiment_path / 'figs'

    atted_score = pd.read_parquet(data_params['atted_path'])
    atted_score = atted_score.set_index(atted_score.columns[0])
    # Make atted symmetric
    a = squareform(atted_score, checks=False)
    a = squareform(a)
    atted_score = pd.DataFrame(a,
                               index=atted_score.index,
                               columns=atted_score.columns)
    local_dist = expr_mat_time.get_distance_matrix(absolute_dist=False)
    # Merge them to be the same set
    selected_genes = atted_score.index.intersection(local_dist.index)
    local_dist = local_dist.loc[selected_genes, selected_genes]
    atted_score = atted_score.loc[selected_genes, selected_genes]
    # Save full dists
    local_dist.to_parquet(experiment_path
                          / 'full_datasets' / 'local_dists.parquet.gzip',
                          compression='gzip')

    # And combined dists
    for w in [0.25,0.5,0.75]:
        sum_dist_df = combine_local_distance_and_prior(local_dist, atted_score,
                                                       dists_out_path=(
                                                                   experiment_path
                                                                   / 'full_datasets' / f"combined_sum_dists_w_{str(w).replace('.','_')}.parquet.gzip"),
                                                       combo='sum',
                                                       # plot_out_path=fig_folder,
                                                       w_global=w
                                                       )

    atted_dist_df = atted_score.max().max() - atted_score
    atted_dist_df.to_parquet(experiment_path
                             / 'full_datasets' / 'atted_dists.parquet.gzip',
                             compression='gzip')
    return atted_dist_df, atted_score, experiment_path, local_dist, sum_dist_df


def drought_data_to_sbml(experiment_path):
    # Load the config file
    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)
    expr_mat_time = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])

    my_ode = from_expr_mat_time_to_ode(data_params, experiment_path,
                                       expr_mat_time, hyper_params)
    u_t_function = 'drought * time / 13'
    logging.info(my_ode)
    my_ode.save_to_sbml(experiment_path / 'module_network.xml', u_t_function)

def analyse_go_enrichments_find_enrichment(
        in_path: Path,
        out_path: Path,
        ax_to_plot_on: bool | plt.Axes = False,
        do_annotations: bool = True):
    # For DS and Method
    all_result_df = read_go_enrich_files_into_df(in_path)
    # plot_gene_modules_ds_size_distribution(all_result_df, out_path)
    valid_rows = extract_only_selected_ds_row_from_df(all_result_df, in_path)
    # valid_rows = all_result_df

    # Main figures
    # Fraction of modules with > 0 GO term
    at_least_one_go_term_barplot_keywords = dict(
        data=valid_rows, y='nr_enriched_go_terms', x='method',
        estimator=lambda y: (y > 0).sum() / len(y),
        order=['atted_dists', 'combined_sum_dists_w_0_75', 'combined_sum_dists_w_0_5', 'combined_sum_dists_w_0_25', 'local_dists']
    )
    # plt.close()
    if ax_to_plot_on:
        ax = sns.barplot(ax=ax_to_plot_on, **at_least_one_go_term_barplot_keywords)
    else:
        ax = sns.barplot(**at_least_one_go_term_barplot_keywords)
    ax.set_ylabel('Fraction of modules with > 0 enriched GO term')

    if do_annotations:
        pairs = list(combinations(
            ['atted_dists', 'combined_sum_dists', 'local_dists'],
            # ['atted_dists', 'combined_sum_dists', 'local_dists', 'random'],
            2))

        annotator = Annotator(
            ax, pairs, **at_least_one_go_term_barplot_keywords
        )
        annotator.configure(test='Mann-Whitney',
                            loc='outside')
        annotator.apply_and_annotate()
    if ax_to_plot_on:
        return
    # plt.tight_layout()
    ax.tick_params(axis='x', labelrotation=90)
    plt.savefig(out_path / 'fraction_at_least_one_go_term_selected_ds.svg',
                bbox_inches='tight')
    plt.close()

    # GO semantic similarity scores
    ax =  sns.boxplot(data=valid_rows, y='semantic_similarity', x='method')
    if do_annotations:
        annotator.new_plot(ax, pairs, data=valid_rows, y='semantic_similarity',
                           x='method')
        annotator.apply_and_annotate()
    plt.savefig(out_path / 'semantic_similarity_boxplot.svg',
                bbox_inches='tight')
    plt.close()

    # Other figures
    ax = sns.boxplot(data=valid_rows, y='nr_enriched_go_terms', x='method')
    if do_annotations:
        annotator.new_plot(ax, pairs, data=valid_rows, y='nr_enriched_go_terms',
                              x='method')
        annotator.apply_and_annotate()
    plt.savefig(out_path /
                'go_terms_per_module_boxplot_selected_ds.svg', bbox_inches='tight')
    plt.close()

    # Of these GO-terms, what is their semantic similarity?
    sns.scatterplot(data=valid_rows, x='nr_enriched_go_terms',
                    y='semantic_similarity', hue='method')
    plt.savefig(out_path / 'scatterplot_enriched_go_terms_semantic_sim_selected_ds.svg')
    plt.close()

    sns.jointplot(data=valid_rows, x='nr_enriched_go_terms',
                    y='semantic_similarity', hue='method')
    plt.savefig(out_path / 'jointplot_enriched_go_terms_semantic_sim_selected_ds.svg')
    plt.close()

def from_list_create_records(param_value_list, hill_order, exp_name):
    if exp_name == 'drought':
        petab_yaml_path = 'data/experiments/30_response_to_reviewers/hill_order_1/drought/petab_files/baddadan_drought_petab.yaml'
    elif exp_name == 'heat':
        petab_yaml_path = 'data/experiments/30_response_to_reviewers/hill_order_1/heat/petab_files/baddadan_heat_petab.yaml'
    else:
        raise ValueError
    petab_problem = petab.v1.Problem.from_yaml(petab_yaml_path)
    assert len(param_value_list) == len(petab_problem.parameter_df)
    for param_name, param_value in list(zip(petab_problem.parameter_df.index.to_list(),
             param_value_list)) :
        yield (exp_name, hill_order, param_name, param_value)

def compare_parameters_between_hill_orders():

    records = []
    # Just copy-pasted the params here to save time

    hill_order = 1
    exp_name = 'drought'
    param_value_list = [0.00000000e+00, -3.34941862e-01, -4.85531338e+00, 7.25557267e-01,
        -1.72586631e+01, -3.86700540e+00, 2.39113658e+00, 1.00000000e+00,
        -2.55340546e+01, 9.30899389e+00, 2.29165029e+00, 6.78925618e-01,
        -5.55722853e+00, 5.28298540e-01, -1.35096428e+01, 9.77602257e+00,
        2.21877157e+00, 9.97501840e-01, -2.27847827e+00, 5.74916965e-01,
        -4.23428857e+00, 2.64086135e-01, -7.04802270e-04, 4.35045553e-01,
        -1.67669224e+01, 2.01531630e+00, -5.16538074e+00, -8.64098502e-01,
        -3.69029045e+00, 9.24518853e-01, 2.36545856e+00, 9.99976092e-01,
        -3.60960482e+00, -9.43468800e-01, -1.86288623e+01, 1.00000000e+01,
        2.03385372e+00, 4.80261581e-01, -2.38847650e+01, 1.00000000e+01,
        2.48251368e+00, 1.00000000e+00, -4.81361541e+00, 9.12578217e-01,
        -1.37879546e+00, 1.00000000e+00, -2.06807980e+00, 3.78704283e-01,
        -2.64346732e+00, -5.92304194e-01]

    records.extend(from_list_create_records(param_value_list, hill_order, exp_name))

    exp_name = 'heat'
    param_value_list = [-0.5460262, 1.70805759, -5.45258915, -0.23220054, 1.98832469,
        -0.95116928, -5.6536398, -2.43177523, -2.75148761, -2.47960166,
        -1.91303155, -3.36929136, 0.40359694, -4.05801035, 0.38664217,
        0.53964787, 3.14159265, 10.0, -49.39505545, 0.59470926,
        2.39674894, 0.59575701, -5.32799333, 0.53041615, 1.78734318,
        0.88743527, -6.3479198, -2.64811188, -2.4672032, -5.78330414,
        0.47980794, 0.16787094, -2.74442424, 9.97741353, -1.62311173,
        0.91188365, 1.25220268, 0.9991174, -4.3906762, -0.9990819,
        0.10776662, 0.87727807, -9.99998109, -12.29955092, 9.28864236,
        2.96629495, -1.0, -5.00578728, 0.2659731, -2.60959135,
        -1.45297931, 0.38470963, -47.97141547, -1.29236776, 2.83837837,
        1.0, -4.04695641, -0.9175726, -10.0, -23.91303908,
        2.90611837, -5.64875494, 0.46454112, 2.99963561, -0.68116389,
        -4.57938569, 0.20747781, 1.47702414, -4.01146759, 6.27989802,
        0.33445832, -2.88857643, 10.0]

    records.extend(
        from_list_create_records(param_value_list, hill_order,
                                 exp_name))
    hill_order = 2
    exp_name = 'drought'
    param_value_list = [-1.29829257e+01, -4.13400768e+00, 1.89479277e+00, 4.77624262e-01,
        -4.40884177e+00, -8.02892555e+00, 1.67567615e+00, 5.98415442e-01,
        -4.86279644e+01, 7.29414592e+00, 2.40902849e+00, 4.09462398e-01,
        -5.85135105e+00, 4.98410744e-02, -1.71352345e+01, 4.63937281e+00,
        2.02670050e+00, 4.89787898e-01, -5.26442633e+00, 9.89352259e-01,
        -5.87285700e+00, 6.83111833e-01, 0.00000000e+00, 4.30992385e-01,
        -4.81118466e+01, -1.00000000e+01, -6.00000000e+00, 9.98366033e-01,
        1.82083525e+00, -1.00000000e+00, -5.35541302e+00, 9.79239477e-01,
        2.97956231e+00, 1.00000000e+00, -1.59787900e-02, 3.61172899e-01,
        -5.99965365e+00, 9.99482840e-01, -3.63533001e+01, -9.71477066e+00,
        -5.19454897e+00, 2.98537327e-01, 1.62321922e+00, -1.00000000e+00,
        -5.82667687e+00, 1.32615786e-01, -2.37261884e+00, 6.76363853e-01,
        2.97450369e+00, 1.00000000e+00]

    records.extend(
        from_list_create_records(param_value_list, hill_order,
                                 exp_name))

    exp_name = 'heat'
    param_value_list = [-35.68702996, 7.29541046, 2.43465553, 0.42079014, -5.67245029,
        -0.99999998, 1.24971416, -3.14159265, -9.999639, -1.60372074,
        -1.28725875, -3.37514782, -0.79147651, -2.81430989, 0.13881748,
        0.34852835, 3.14159265, 6.52139167, -5.58885348, 3.75822383,
        -2.4037046, 1.0, 1.0668709, -1.0, 0.84142191,
        0.69717016, -1.96931112, -4.77782451, -4.93511529, 1.3097554,
        0.03877861, -4.25581064, 1.04209152, -0.70703833, -3.18807585,
        1.64055782, -3.99688115, -0.71651645, 1.32000768, 0.10684638,
        0.37993437, 0.71107932, -10.0, -28.98100143, -10.0,
        -3.24130932, 0.81595611, 2.29592358, 0.54261054, 1.20210577,
        2.35760875, 8.62466979, -3.40795353, -1.29584235, 2.04417025,
        1.0, 0.68525122, 3.14159265, 3.766625, -0.2951119,
        2.04973664, 1.55764656, 0.89647388, -3.14411046, 0.75311115,
        0.53988612, -0.25420156, -6.63449903, -1.07320478, 1.75268803,
        -0.72960853, 2.74419591, 2.4884024]

    records.extend(
        from_list_create_records(param_value_list, hill_order,
                                 exp_name))
    hill_order = 3
    exp_name = 'drought'
    param_value_list = [-4.89565134e+01, -9.99543844e+00, 2.32851525e+00, 6.00440428e-01,
        -4.53434131e-01, -1.22281955e+00, 6.56329467e-01, 4.93629855e-01,
        -4.42478133e+01, 1.00000000e+01, 2.28189597e+00, 3.51707102e-01,
        -4.72433005e+00, -9.38559334e-01, -1.54192528e+01, 1.00000000e+01,
        1.57832067e+00, -1.00000000e+00, -5.64644141e+00, -6.53817815e-01,
        2.38169087e+00, 1.00000000e+00, -3.53566989e-03, 4.54052054e-01,
        -8.16831613e+00, -1.00000000e+01, -4.07780547e+00, -9.74362150e-01,
        1.85163321e+00, 5.44770651e-01, -5.05412987e+00, -5.69645756e-01,
        -5.94178717e+00, 9.63195107e-01, -2.53829112e+01, 1.00000000e+01,
        1.99401148e+00, 2.98440307e-01, -4.22423674e+00, -9.16848437e+00,
        -2.43583693e+00, -8.84977958e-01, -2.52209923e+00, 1.84403518e-02,
        -2.95253866e+00, 9.41945836e-01, 1.88057070e+00, 6.84939572e-01,
        -4.17895723e+00, -2.74345544e-01]

    records.extend(
        from_list_create_records(param_value_list, hill_order,
                                 exp_name))

    exp_name = 'heat'
    param_value_list = [0.00000000e+00, 3.93008411e-02, -1.95754090e+00, -6.37778912e-01,
        -3.64285497e+00, 5.44581517e-01, -3.00486252e-01, 1.63806777e+00,
        4.50451654e-02, -4.48864805e+01, -6.93944995e+00, 2.93336746e+00,
        -8.73933398e-02, 2.07488608e+00, 1.05358343e-01, -3.92765277e+00,
        9.40531549e-01, 9.77449476e+00, -4.08752340e+00, 2.66248369e+00,
        -1.53211975e+00, 7.75706555e-01, -3.50886659e+00, 4.44548186e-01,
        7.04152379e-01, 7.37339140e-01, 7.17125316e+00, -1.04898116e+01,
        -9.80710001e+00, 1.68828359e+00, 1.20869348e-01, -3.15936327e+00,
        2.10062303e+00, -6.87420788e+00, -2.78127425e+01, 9.58260013e+00,
        2.06521047e+00, 5.62047637e-01, -5.82314918e+00, 5.62495892e-01,
        -5.03237892e+00, -4.28255933e-01, 7.32665855e-01, -4.00728226e+00,
        -2.86286781e+00, -5.38930486e+00, -4.44859569e-01, 9.64607402e-01,
        4.48297896e-01, -4.82523483e+00, -5.26254686e-01, 8.85140061e+00,
        -6.37890214e-02, 4.10942055e-02, -4.78146498e+00, 7.44927473e-01,
        -6.30592050e-01, -3.14159265e+00, 1.10072871e-01, -2.84951646e+01,
        1.00000000e+01, 2.19897702e+00, 5.23988899e-01, -4.66800086e+00,
        8.63902423e-01, -4.33855199e+00, -1.28946919e+00, 1.00000000e+01,
        -1.70109394e+00, 2.70768982e+00, -4.12816990e+00, -1.64010371e-01,
        3.91038703e+00]

    records.extend(
        from_list_create_records(param_value_list, hill_order,
                                 exp_name))
    out_dir = Path('data/experiments/30_response_to_reviewers')
    # Create the DataFrame (assuming 'records' is already defined)
    df = pd.DataFrame.from_records(records, columns=['exp_name', 'hill_order',
                                                     'param_name',
                                                     'param_value'])

    # Separate the parameters into linear and non-linear categories
    linear_params = ['delta', 'gamma', 'b_', 'phi']
    nonlinear_params = ['beta', 'k', 'a']

    # Filter the data for each category
    df_linear = df[df['param_name'].str.startswith(tuple(linear_params))]
    df_nonlinear = df[df['param_name'].str.startswith(tuple(nonlinear_params))]

    # Pivot the data for each category
    df_linear_pivot = df_linear.pivot(index=["exp_name", "param_name"],
                                      columns="hill_order",
                                      values="param_value").reset_index()
    df_linear_pivot['param_name'] = df_linear_pivot.param_name.apply(lambda x: x.split('_')[0])
    df_linear_pivot.columns = ["exp_name", "param_name", "hill_1", "hill_2",
                               "hill_3"]

    df_nonlinear_pivot = df_nonlinear.pivot(index=["exp_name", "param_name"],
                                            columns="hill_order",
                                            values="param_value").reset_index()
    df_nonlinear_pivot['param_name'] = df_nonlinear_pivot.param_name.apply(
        lambda x: x.split('_')[0])
    df_nonlinear_pivot.columns = ["exp_name", "param_name", "hill_1", "hill_2",
                                  "hill_3"]
    df_nonlinear_pivot.loc[:, 'hill_1':'hill_3'] = df_nonlinear_pivot.loc[:, 'hill_1':'hill_3'].map(lambda x: 10 ** x)

    df_final = pd.concat([df_nonlinear_pivot, df_linear_pivot])

    # Hill 1 vs 2
    g = sns.relplot(data=df_final, col='exp_name', row='param_name', x='hill_2', y='hill_1', facet_kws={'sharex': False, 'sharey': False,})
    corr_df = df_final.groupby(['param_name', 'exp_name'])[["hill_1", "hill_2"]].corr(method='pearson')
    # Iterate through the subplots and add correlation text
    for (param_name, exp_name), ax in g.axes_dict.items():
        # Extract the correlation value
        try:
            corr_value = corr_df.xs((param_name, exp_name)).loc[
                "hill_1", "hill_2"]
            ax.text(
                0.05, 0.95,
                f"r = {corr_value:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle="round", alpha=0.1)
            )
        except KeyError:
            # Handle missing data
            ax.text(
                0.05, 0.95,
                "r = N/A",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle="round", alpha=0.1)
            )
    plt.tight_layout()
    plt.savefig(out_dir / 'all_params_hill_2_vs_hill_1.svg')
    plt.show()

    # Hill 2 vs 3
    g = sns.relplot(data=df_final, col='exp_name', row='param_name', x='hill_2', y='hill_3', facet_kws={'sharex': False, 'sharey': False,})
    corr_df = df_final.groupby(['param_name', 'exp_name'])[["hill_3", "hill_2"]].corr(method='pearson')
    # Iterate through the subplots and add correlation text
    for (param_name, exp_name), ax in g.axes_dict.items():
        # Extract the correlation value
        try:
            corr_value = corr_df.xs((param_name, exp_name)).loc[
                "hill_3", "hill_2"]
            ax.text(
                0.05, 0.95,
                f"r = {corr_value:.2f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle="round", alpha=0.1)
            )
        except KeyError:
            # Handle missing data
            ax.text(
                0.05, 0.95,
                "r = N/A",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle="round", alpha=0.1)
            )
    plt.tight_layout()
    plt.savefig(out_dir / 'all_params_hill_2_vs_hill_3.svg')
    plt.show()


def extract_only_selected_ds_row_from_df(all_result_df, in_path):
    """Do not compare between all different deepsplit (DS) values
    (hyperparameter of WGCNA), but only select one DS for each method
    to ensure that they are all the same size
    """
    # Keep only to one comparison
    # Select deepsplit value on most comparable module sizes
    if np.all(all_result_df['method'] == 'random'):
        valid_rows = all_result_df
    elif 'drought' == in_path.parent.name:
        valid_rows = all_result_df[
            (all_result_df['method'].isin(['local_dists', 'combined_sum_dists_w_0_25'])
            & (all_result_df['deepsplit'] == '2'))
            | (all_result_df['method'].isin(
                ['atted_dists', 'combined_sum_dists_w_0_5', 'combined_sum_dists_w_0_75', 'random']))
            & (all_result_df['deepsplit'] == '1')
            ]
    elif 'heat' == in_path.parent.name:
        valid_rows = all_result_df[
            (all_result_df['deepsplit'] == '1')
            & (all_result_df['method'].isin(
                ['atted_dists', 'combined_sum_dists_w_0_25', 'combined_sum_dists_w_0_5', 'combined_sum_dists_w_0_75', 'local_dists',
                 'random']))
            ]
    assert len(valid_rows) > 0
    return valid_rows


def plot_gene_modules_ds_size_distribution(all_result_df: pd.DataFrame, out_path: Path, do_annotations = True):

    # Remove the random clustering because not relevant for this (its size distribution is equal to the combined_sum_dists)
    all_result_df = all_result_df[all_result_df['method'] != 'random']
    # if 'drought' in out_path.parts:
    #     size_pairs.extend(
    #         (
    #             (('2', 'local_dists'), ('1', 'combined_sum_dists')),
    #             (('2', 'local_dists'), ('1', 'atted_dists')),
    #         )
    #     )
    # Size distributions
    ax = sns.boxplot(
        data=all_result_df, x='module_size', hue='method',
        y='deepsplit'
    )
    if do_annotations:
        size_pairs = [
            (('1', 'atted_dists'), ('1', 'combined_sum_dists')),
            (('1', 'local_dists'), ('1', 'combined_sum_dists')),
            (('1', 'atted_dists'), ('1', 'local_dists')),
            # (('2', 'atted_dists'), ('2', 'combined_sum_dists')),
            # (('2', 'local_dists'), ('2', 'combined_sum_dists')),
            # (('2', 'atted_dists'), ('2', 'local_dists')),
        ]
        annotator = Annotator(
            ax, size_pairs, data=all_result_df, x='module_size',
            hue='method', y='deepsplit', orient='h'
        )
        annotator.configure(test='Mann-Whitney',
                            loc='outside', text_format='star')
        annotator.apply_and_annotate()
    plt.savefig(out_path / 'module_sizes_deepsplit_hue_is_method_boxplot.svg',
                bbox_inches='tight')
    plt.close()


def read_go_enrich_files_into_df(in_path):
    out_records = []
    for file in tqdm(list(in_path.glob('*.log'))):
        with file.open('r') as f:
            text = f.read()
            module_size = re.search('(?<=Study:) \d+', text).group()
        module_size = int(module_size)

        if 'wgcna' in file.name:
            method = file.name.split('_wgcna')[0]
        else:
            method = 'random'
        deepsplit_value = re.search('(?<=ds)\d+', file.name).group()
        module_nr = re.search('(?<=module_)\d+', file.name).group()
        go_output_file_name = file.with_suffix('.tsv')
        if go_output_file_name.exists():
            one_module_df = pd.read_csv(go_output_file_name, sep='\t')
            go_terms = EnrichedGeneModuleGoTerms(one_module_df)
            semantic_similarity = go_terms.overall_wang_similarity()
            nr_enriched = go_terms.get_nr_go_terms()
        else:
            nr_enriched = 0
            semantic_similarity = np.nan
        out_records.append(
            (method, deepsplit_value, module_nr, nr_enriched,
             module_size, semantic_similarity)
        )
    all_result_df = pd.DataFrame.from_records(
        out_records,
        columns=['method', 'deepsplit', 'module_id',
                 'nr_enriched_go_terms', 'module_size', 'semantic_similarity']
    )
    # Min dists weren't too good so leave them out of the analysis
    all_result_df = all_result_df[all_result_df['method'] != 'combined_min_dists']
    return all_result_df


def heat_data_to_sbml(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    expr_mat_time = expr_mat_from_heat(data_params['in_path'],
                                       hyper_params['agg_method'],
                                       hyper_params['do_log2'])

    my_ode = from_expr_mat_time_to_ode(data_params, experiment_path,
                                       expr_mat_time, hyper_params)

    # These are parameters that are different between the two datasets
    u_t_function = 'temp'
    my_ode.save_to_sbml(experiment_path / 'module_network.xml', u_t_function)


def from_expr_mat_time_to_ode(data_params,
                              experiment_path: Path,
                              expr_mat_time: ExpressionMatrixTimeSeries,
                              hyper_params):
    wgcna_module_assignment = data_params['wgcna_module_assignment_path']
    expr_mat_time.assign_clusters_from_wgcna(wgcna_module_assignment)
    # logging.warning('New clusters so new TF2Network analysis?')
    tf2_in_path = experiment_path / '03_tf2_input.txt'
    tf2_out_path = experiment_path / '04_tf2network_output.tsv'
    # Post to tf2network
    expr_mat_time.write_tf2_input_file(
        out_path=tf2_in_path)

    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=hyper_params['edge_corr_threshold'],
        module_plot_path=experiment_path / 'no_filter_global_cluster_module_network.svg')
    print('Unfiltered network')
    module_module.print_stats()

    expr_mat_time.keep_highest_z_clusters(
        hyper_params['top_nr_clusters'],
        tf2_output_path=tf2_out_path,
        plotting_path=experiment_path / 'figs')
    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=hyper_params['edge_corr_threshold'],
        module_plot_path=experiment_path / 'filtered_global_cluster_module_network.svg')
    print('Filtered network')
    module_module.print_stats()

    expr_mat_time.keep_only_modules_in_network(module_module)
    # expr_mat_time.plot_clusters_over_time()
    with (experiment_path / 'expr_mat_time.pkl').open('wb') as f:
        pickle.dump(expr_mat_time, f)
    with (experiment_path / 'module_network.pkl').open('wb') as f:
        pickle.dump(module_module, f)
    module_module.save_for_cytoscape(experiment_path / 'module_network.cyjs')
    # # Mean number of TFs for each edge:
    # pd.Series([len(i)
    #            for (_,_,i) in module_module.graph.edges(data='tf_name')]
    #           ).value_counts(normalize=True)

    assert expr_mat_time.has_been_clustered
    # expr_mat_time.get_genes_per_cluster()[328]
    my_ode = OdeModel.construct_from_regulatory_network(
        module_module,
        nonlinear=True,
        add_circadian_clock=hyper_params['add_circadian_clock'],
        n=hyper_params['hill_equation_order'])
    return my_ode


def pypesto_from_sbml(experiment_path: Path,
                      condition: str,
                      expr_mat_time_pkl_path: Path,
                      sbml_path: Path,
                      do_ml_flow_logging: bool = True,
                      use_best_params_as_init: Path | None = None):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path
    )
    with expr_mat_time_pkl_path.open('rb') as f:
        expr_mat_time: ExpressionMatrixTimeSeries = pickle.load(f)

    # Add constant value
    expr_mat_time.add_constant(3, do_all_pos_check=False)
    # expr_mat_time.get_ci_per_cluster()
    param_guess_dict = None
    if use_best_params_as_init:
        logging.info('Using previously found parameters')
        # Read result from hdf5
        loaded_result = pypesto.store.read_result(use_best_params_as_init)
        # Match param names to param values
        param_guess_dict = dict(zip(loaded_result.problem.x_names,
                                    loaded_result.optimize_result[0].x))

    write_petab_files(
        expr_mat_time,
        sbml_path,
        experiment_path / 'petab_files',
        experimental_setup=condition,
        do_interpolate=hyper_params['do_interpolate'],
        do_extra_datapoints=hyper_params['do_extra_datapoints'],
        param_guess_dict=param_guess_dict
    )
    # Experimental data
    petab_problem = petab.v1.Problem.from_yaml(
        str(experiment_path / 'petab_files' / f'baddadan_{condition}_petab.yaml')
    )

    if hyper_params['do_sim_data']:
        petab_problem = simulated_data_pypesto(
            petab_problem,
            experiment_path /  'amici_models' / 'baddadan_sim'
        )
    # sns.relplot(data=petab_problem.measurement_df, y='measurement', x='time',
    #             col='observableId', hue='simulationConditionId', col_wrap=4,
    #             kind='line');
    # plt.show()
    # DO experiment on real data
    result = param_optimise_petab_problem(
        petab_problem, experiment_path, n_starts=hyper_params['n_starts']
    )

    # result = pypesto.store.read_result(
    #     experiment_path / 'pypesto_results.hdf5', optimize=True
    # )
    # Get data from best run
    result_dict = result.optimize_result.as_dataframe().iloc[0, :].to_dict()

    param_values = {param_name: param_value for param_name, param_value
                    in zip(result.problem.x_names, result_dict['x'])}
    if do_ml_flow_logging:
        with mlflow.start_run(
                description=experiment_params['description']):
            mlflow.log_params(data_params)
            mlflow.log_params(hyper_params)
            mlflow.set_tags(experiment_params)
            mlflow.log_artifact(str(experiment_path))
            mlflow.log_params(param_values)
            # Log fval as metric
            mlflow.log_metric('fval', result_dict['fval'])
            # mlflow.log_artifact(
            #     str(experiment_path / 'figures'))

def simulated_data_pypesto(petab_problem, model_folder):
    # Do simulated data
    # model_folder = model_folder / 'amici_models' / 'baddadan'
    importer = pypesto.petab.PetabImporter(petab_problem,
                                           simulator_type="amici",
                                           output_folder=str(model_folder)
                                           )
    factory = importer.create_objective_creator()
    obj = factory.create_objective()
    # SIMULATED DATAAAAAAAAAAA
    petab_problem_synthetic = copy.deepcopy(petab_problem)
    simulation_param_dict = {}
    for param_name in petab_problem_synthetic.parameter_df.index:
        if param_name == 'delta_0':
            value = -.1
        elif 'delta' in param_name:
            value = -.1
        # elif param_name == 'gamma_0':
        #     value = 3
        elif 'gamma' in param_name:
            value = .5
        # elif 'beta_0_1' in param_name:
        #     value = .01
        # elif 'k_1_2' == param_name:
        #     value = 6
        elif 'beta' in param_name:
            value = 10
        # elif 'k_0_1' == param_name:
        #     value = 1
        elif param_name.startswith('k_'):
            value = 2
        else:
            raise NotImplementedError
        simulation_param_dict[param_name] = value
    # expr_mat_time.
    # obj.amici_model.setInitialStates([1, 2, 3])
    petab_problem_synthetic.parameter_df[petab.v1.C.NOMINAL_VALUE] \
        = petab_problem_synthetic.parameter_df.index.map(
        simulation_param_dict
    )
    # petab_problem_synthetic.parameter_df['estimate'] = 0
    simulator = amici.petab.simulator.PetabSimulator(petab_problem_synthetic)
    petab_problem_synthetic.measurement_df = simulator.simulate(
        noise=True,
        # noise_scaling_factor=0.01,
        # Optional: the AMICI simulator is provided a model, to avoid recompilation
        amici_model=obj.amici_model,
        as_measurement=True,
    )
    simulator.remove_working_dir()
    sns.lineplot(data=petab_problem_synthetic.measurement_df, y='measurement',
                 x='time',
                 hue='observableId', style='simulationConditionId')
    plt.show()
    # sns.lineplot(data=petab_problem.measurement_df, y='measurement', x='time',
    #              hue='observableId', style='simulationConditionId')
    # plt.show()
    return petab_problem_synthetic


def do_coherence_with_stat_tests(in_dir: Path,
                                 expr_mat_time: ExpressionMatrixTimeSeries,
                                 out_dir: Path,
                                 ax_to_plot_on: bool | plt.Axes = False,
                                 do_stat_test: bool = True):
    """Measure coherence between different clusterings and do statistical test"""
    out_records = []
    for method in ['atted_dists', 'combined_sum_dists_w_0_25', 'combined_sum_dists_w_0_5', 'combined_sum_dists_w_0_75',
                   'local_dists']:
        for ds_filename in ['ds1']: #, 'ds2']:
            # if (method, ds_filename) == ('random', 'ds2'):
            #     continue
            if 'drought' in in_dir.parts and (method in ['local_dists', 'combined_sum_dists_w_0_25']):
                ds_filename = 'ds2'
            expr_mat_time_copy = copy.deepcopy(expr_mat_time)
            pattern = f"{method}*{ds_filename}*"
            files = list(in_dir.glob(pattern))
            assert len(files) > 0, f'No files found for {method} & {ds_filename}'
            expr_mat_time_copy.assign_clusters_from_split_by_module_files(files)
            expr_mat_time_copy.do_z_scaling()
            module_coherences = expr_mat_time_copy.get_all_explained_vars()
            ds_value = ds_filename.split('ds')[1]
            out_records.extend(
                [
                    [method, ds_value, coherence]
                    for coherence in module_coherences]
            )

    df = pd.DataFrame.from_records(out_records,
                                   columns=['method', 'deepsplit', 'coherence']
                                   )
    # df = extract_only_selected_ds_row_from_df(df, in_dir)
    if ax_to_plot_on:
        ax = sns.boxplot(data=df, y='coherence', x='method', ax=ax_to_plot_on)
    else:
        ax = sns.boxplot(data=df, y='coherence', x='method', order = ['atted_dists', 'combined_sum_dists_w_0_75', 'combined_sum_dists_w_0_5', 'combined_sum_dists_w_0_25', 'local_dists'])

    if do_stat_test:
        pairs = list(combinations(
            # ['atted_dists', 'combined_sum_dists', 'local_dists'],
            ['atted_dists', 'combined_sum_dists', 'local_dists', 'random'],
            2))

        annotator = Annotator(
            ax, pairs, data=df, y='coherence', x='method'
        )
        annotator.configure(test='Mann-Whitney',
                            loc='outside')
        annotator.apply_and_annotate()
    if ax_to_plot_on:
        return
    plt.ylim((0, .9))
    ax.tick_params(axis='x', labelrotation=90)
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / 'boxplot_coherence_with_stat_test.svg',
                bbox_inches='tight')
    plt.close()


def plot_module_size_distributions(pkl_path: Path):
    with pkl_path.open('rb') as f:
        cluster_dict = pickle.load(f)
    records = []
    for dist, expr_mat in cluster_dict.items():
        sizes = expr_mat.df.cluster_id.value_counts()
        for size in sizes:
            records.append((dist, size))
    df = pd.DataFrame.from_records(records, columns=['method', 'size'])
    sns.boxplot(data=df, y='size', x='method')
    plt.title(pkl_path.name.split('_')[0])
    plt.yscale('log')
    plt.ylabel('Module size')
    plt.show()
    df.groupby('method')['size'].sum()


def combine_local_distance_and_prior(local_dist: pd.DataFrame,
                                     prior_score: pd.DataFrame,
                                     dists_out_path: Path, combo: str = 'sum',
                                     plot_out_path: Path | None = None,
                                     w_global: float = 0.5):
    """Get combo of local distances and atted_distances to
    do distances simulatenously

    :param local_dist: local distances as dataframe
    :param prior_score: prior scores (e.g. from atted)
    :param dists_out_path: file path to save combined distances (should be a name that ends in .parquet.gzip)
    :param combo: how to combine the distances (sum vs minimum of either)
    :param plot_out_path: directory in which to save figures. If none, no plotting
    :return:
    """

    # toy_size = 5000
    # atted_score = atted_score.iloc[:toy_size, :toy_size]
    # local_dist = local_dist.iloc[:toy_size, :toy_size]

    # get intersection
    selected_genes = prior_score.index.intersection(local_dist.index)
    # Shrink dataframes so match in size
    local_dist = local_dist.loc[selected_genes, selected_genes]
    prior_score = prior_score.loc[selected_genes, selected_genes]

    # Higher score -> lower dist
    atted_dist = prior_score.max().max() - prior_score

    assert (local_dist.index.equals(
        atted_dist.index) and local_dist.columns.equals(atted_dist.columns))

    local_dist_flat = squareform(local_dist)
    atted_dist_flat = squareform(atted_dist, checks=False)

    # Convert into z-scores
    local_dist_flat_norm = (local_dist_flat - np.mean(
        local_dist_flat)) / np.std(local_dist_flat)
    atted_dist_flat_norm = (atted_dist_flat - np.mean(
        atted_dist_flat)) / np.std(atted_dist_flat)

    if plot_out_path:
        # Plot both distributions
        sns.histplot(local_dist_flat, binwidth=.2, element='step',
                     fill=False, common_norm=False)
        sns.histplot(atted_dist_flat, binwidth=.2, element='step',
                     fill=False, common_norm=False)
        plt.legend(['Local Distances', 'Atted Distances'])

        plt.savefig(plot_out_path / 'raw_input_distances.png')
        plt.close()

        sns.histplot(local_dist_flat_norm, binwidth=.2, element='step',
                     fill=False, common_norm=False)
        sns.histplot(atted_dist_flat_norm, binwidth=.2, element='step',
                     fill=False, common_norm=False)
        plt.legend(['Local Distances', 'Atted Distances'])
        plt.savefig(plot_out_path / 'normalised_distances.png')
        plt.close()

    if combo =='sum':
        combined_distances = ((1 - w_global) * local_dist_flat_norm
                              + w_global * atted_dist_flat_norm)
    elif combo == 'min':
        combined_distances = np.minimum(local_dist_flat_norm,
                                        atted_dist_flat_norm)
        if plot_out_path:
            sns.scatterplot(y=atted_dist_flat_norm, x=local_dist_flat_norm, s=.2)
            # make line
            x = np.linspace(min(local_dist_flat_norm), max(local_dist_flat_norm))
            plt.plot(x, x, color=sns.color_palette()[1])
            plt.xlabel('Local')
            plt.ylabel('Atted')
            plt.savefig(plot_out_path / 'scatter_atted_vs_local.png')
    else:
        raise NotImplementedError

    # sns.histplot(combined_distances, binwidth=.2, element='step', fill=False)
    # plt.savefig(out_path / 'combined_distances.png')
    # plt.close()

    # Rescale and reshape again
    combined_distances = combined_distances - np.min(combined_distances)
    combined_distances = combined_distances / max(combined_distances)
    if plot_out_path:
        sns.histplot(combined_distances)
        plt.savefig(plot_out_path / 'combined_distances.png')
        plt.close()

    square_no_negative = squareform(combined_distances)
    combined_dist_df = pd.DataFrame(data=square_no_negative,
                                  index=local_dist.index,
                                  columns=local_dist.index)

    combined_dist_df.to_parquet(dists_out_path, compression='gzip')

    # / f'atted_local_dist_{combo}_combined_no_negative.parquet.gzip'
    return combined_dist_df


def get_coherence_random_modules(wgcna_label_file : str,
                                 expr_mat_time: ExpressionMatrixTimeSeries,
                                 figure_out_dir: Path):
    expr_mat_time.assign_clusters_from_wgcna(wgcna_label_file)
    coherence_entry = expr_mat_time.get_all_explained_vars()
    coherence_entry = [['combined_dists', i] for i in coherence_entry]

    for i in range(4):
        expr_mat_time_random = copy.deepcopy(expr_mat_time)
        # expr_mat_time_random.do_random_clustering_with_given_size_dist(
        #     data_params['wgna_label_file'])

        expr_mat_time_random.do_random_clustering_with_given_size_dist(
            wgcna_label_file=None,
            use_own_clustering=True
        )
        # Do coherence per module
        coherence_entry_random_cluster = expr_mat_time_random.get_all_explained_vars()
        coherence_entry_random_cluster = [[f'random_{i}', j] for j in coherence_entry_random_cluster]
        coherence_entry.extend(coherence_entry_random_cluster)

    df = pd.DataFrame.from_records(coherence_entry, columns=['Method', 'Coherence'])
    sns.boxplot(data=df, y='Coherence', x='Method', hue='Method')
    plt.ylim((0, .9))
    plt.savefig(figure_out_dir / 'boxplot_coherence_with_random_modules.png')
    plt.close()

    sns.swarmplot(data=df, y='Coherence', x='Method', hue='Method')
    plt.ylim((0, .9))
    plt.savefig(figure_out_dir / 'swarmplot_coherence_with_random_modules.png')
    plt.close()


def check_correlation_cutoffs_for_intermodular_network(expr_mat_time,
                                                       tf2_in_path,
                                                       tf2_out_path,
                                                       plotting_path):
    out_records = []
    cutoff_values = np.arange(0, 1, 0.05)
    for cutoff_value in cutoff_values:
        # get rid of floating point stuff
        cutoff_value = round(cutoff_value, 2)
        try:
            module_module = module_network_from_tf2_output(
                expr_mat_time, tf2_in_path,
                tf2_out_path,
                threshold=cutoff_value,
                module_plot_path=None
            )
            nr_nodes = module_module.graph.number_of_nodes()
            nr_edges = module_module.graph.number_of_edges()
            nr_interconnected_components = len(
                list(nx.weakly_connected_components(module_module.graph))
            )
            # if .7 < cutoff_value < .85:
            #     # plt.title()
            #     module_module.plot_network(title=str(cutoff_value))
            #     plt.show()
        except RegulatoryDisagreementError as e:
            nr_nodes, nr_edges, nr_interconnected_components = (
                np.nan, np.nan, np.nan)
        out_records.append(
            (cutoff_value, nr_nodes, nr_edges, nr_interconnected_components)
        )
    sparsity_df = pd.DataFrame.from_records(
        out_records, columns=['cutoff', 'nr_nodes', 'nr_edges',
                              'one_interconnected_graph']
    )
    plt.plot(sparsity_df['cutoff'], sparsity_df['nr_nodes'],
             label='nr of nodes')
    plt.plot(sparsity_df['cutoff'], sparsity_df['nr_edges'],
             label='nr of edges')
    has_one_interconnected = sparsity_df[
        sparsity_df['one_interconnected_graph'] == 1]
    disjoint_graphs = sparsity_df[sparsity_df['one_interconnected_graph'] > 1]
    plt.plot(has_one_interconnected['cutoff'],
             has_one_interconnected['nr_nodes'],
             'o', label='Graph with a single weakly connected component')
    plt.plot(disjoint_graphs['cutoff'], disjoint_graphs['nr_nodes'],
             'x', label='Graph with multiple disjoint components')
    plt.xlim((0, 1))
    plt.gca().xaxis.set_major_locator(
        MaxNLocator(integer=False, nbins=10))
    plt.gca().xaxis.set_minor_locator(
        AutoMinorLocator(2))
    plt.legend()
    # plt.title('Heat')
    plt.xlabel('Correlation cutoff')
    plt.tight_layout()
    plt.savefig(plotting_path / 'correlation_cutoff_intermodular_network_stats.svg')
    plt.close()


def sa_drought_over_time():
    in_file = Path('data/raw_data/sa_drought_levels.xlsx')
    metab_df = pd.read_excel(in_file)
    # Take mean over 4 biological samples
    metab_df = metab_df.groupby(['Treatment', 'Time (days)'])['Salicylic acid (SA) '].mean().reset_index()
    # sns.lineplot(data=df, x='Time (days)', y='Salicylic acid (SA) glycoside ', hue='Treatment')
    # plt.show()
    expr_mat_pickl_path = Path('data/experiments/25_everything_including_limma/drought/expr_mat_time.pkl')
    with expr_mat_pickl_path.open('rb') as f:
        expr_mat_time: ExpressionMatrixTimeSeries = pickle.load(f)
    expr_mat_time.add_constant(3, do_all_pos_check=False)
    expressions = expr_mat_time.extract_module_expressions_long_form()
    # expression_list = expr_mat_time.split_series_into_different_conditions(
    #     expressions)

    for selected_cluster_id in [4, 15]:
        subset = expressions[expressions['cluster_id'] == selected_cluster_id]
        subset = subset[subset['condition'] != 'zero']
        subset = subset[['time', 'condition', 'expression']]
        # Assuming the first dataset is in `subset` and the second is in `metab_df`
        # Convert time from timedelta to days
        subset['time_days'] = subset['time'].dt.days
        subset.rename(columns={'condition': 'Treatment'}, inplace=True)
        subset['Treatment'] = subset[
            'Treatment'].str.capitalize()

        subset['Treatment'] = subset['Treatment'].replace(
            {'Control': 'Watered'})

        metab_df.rename(columns={
            'Time (days)': 'time_days',
            'Salicylic acid (SA) ': 'SA_level'
        }, inplace=True)
        merged_df = pd.merge(subset, metab_df,
                             left_on=['Treatment', 'time_days'],
                             right_on=['Treatment', 'time_days'], how='inner')

        correlation = merged_df[['expression', 'SA_level']].corr().iloc[0, 1]
        logging.info(
            f"Correlation between expression of module {selected_cluster_id}"
            f" and SA Levels (r = {correlation:.2f})")
        # Plot the correlation
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=merged_df, x='expression', y='SA_level',
                        hue='Treatment')

        plt.xlabel(f"Gene Module {selected_cluster_id} Mean Expression")
        plt.ylabel("Salicylic Acid Level")
        plt.ylim((0,35))
        plt.legend(title="Condition")
        plt.tight_layout()
        plt.show()


    for metabolite in ['Salicylic acid (SA) ', 'Abscisic acid (ABA) ']:
        sns.lineplot(data=metab_df, x='Time (days)',
                     y=metabolite,
                     hue='Treatment',
                     hue_order=['Watered', 'Drought'],
                     )
        plt.ylabel(f'{metabolite[:-1]}, area under peak LC-ESI-QToF MS')
        plt.show()
