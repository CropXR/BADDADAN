import copy
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
    # # And combined min dists -> not doing these
    # min_dist_df = combine_local_distance_and_prior(
    #     local_dist,
    #     atted_score,
    #     dists_out_path=(experiment_path
    #                     / 'full_datasets' / 'combined_min_dists.parquet.gzip'),
    #     combo='min',
    #     calculate_linkages=False,
    #     plot_out_path=fig_folder,
    # )
    # And combined dists
    sum_dist_df = combine_local_distance_and_prior(local_dist, atted_score,
                                                   dists_out_path=(
                                                               experiment_path
                                                               / 'full_datasets' / 'combined_sum_dists.parquet.gzip'),
                                                   combo='sum',
                                                   plot_out_path=fig_folder)

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
        ax_to_plot_on: bool | plt.Axes = False):
    # For DS and Method
    all_result_df = read_go_enrich_files_into_df(in_path)
    # plot_gene_modules_ds_size_distribution(all_result_df, out_path)
    valid_rows = extract_only_selected_ds_row_from_df(all_result_df, in_path)
    # valid_rows = all_result_df

    # Main figures
    # Fraction of modules with > 0 GO term
    at_least_one_go_term_barplot_keywords = dict(
        data=valid_rows, y='nr_enriched_go_terms', x='method',
        estimator=lambda y: (y > 0).sum() / len(y)
    )
    # plt.close()
    if ax_to_plot_on:
        ax = sns.barplot(ax=ax_to_plot_on, **at_least_one_go_term_barplot_keywords)
    else:
        ax = sns.barplot(**at_least_one_go_term_barplot_keywords)
    ax.set_ylabel('Fraction of modules with > 0 enriched GO term')

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
    plt.savefig(out_path / 'fraction_at_least_one_go_term_selected_ds.svg',
                bbox_inches='tight')
    plt.close()

    # GO semantic similarity scores
    ax =  sns.boxplot(data=valid_rows, y='semantic_similarity', x='method')
    annotator.new_plot(ax, pairs, data=valid_rows, y='semantic_similarity',
                       x='method')
    annotator.apply_and_annotate()
    plt.savefig(out_path / 'semantic_similarity_boxplot.svg',
                bbox_inches='tight')
    plt.close()

    # Other figures
    ax = sns.boxplot(data=valid_rows, y='nr_enriched_go_terms', x='method')
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

    # # Everything linked to module sizes just skipping now # #
    # mean_module_size = all_result_df.groupby(
    #     ['method', 'deepsplit'])['module_size'].mean()
    # mean_enriched_go_terms = all_result_df.groupby(
    #     ['method', 'deepsplit'])['nr_enriched_go_terms'].mean()
    #
    # mean_semantic_similarity = all_result_df.groupby(
    #     ['method', 'deepsplit'])['semantic_similarity'].mean()
    #
    #
    # mean_module_size_and_go_enrichments = pd.concat(
    #     [mean_module_size, mean_enriched_go_terms, mean_semantic_similarity], axis=1).reset_index()
    # mean_module_size_and_go_enrichments = mean_module_size_and_go_enrichments.rename(
    #     {'module_size': 'mean_module_size',
    #         'semantic_similarity': 'mean_semantic_similarity'}, axis='columns')
    #
    # sns.scatterplot(data=mean_module_size_and_go_enrichments, x='mean_module_size',
    #                 y='nr_enriched_go_terms', hue='method', style='deepsplit')
    # plt.ylabel('Mean nr of enriched go terms per module')
    # # Add error bar?
    # plt.savefig(out_path / 'module_size_mean_nr_go_terms_scatterplot.png')
    # plt.show()
    # plt.close()
    #
    # sns.scatterplot(data=mean_module_size_and_go_enrichments, x='mean_module_size',
    #                 y='mean_semantic_similarity', hue='method', style='deepsplit')
    # plt.ylabel('Mean semantic similarity within a module')
    # plt.savefig(out_path / 'module_size_mean_semantic_sim_scatterplot.png')
    # # Add error bar?
    # plt.show()
    # plt.close()
    #
    #
    # mean_module_size = mean_module_size.reset_index()
    # mean_module_size = mean_module_size.rename(
    #     {'module_size': 'mean_module_size'}, axis='columns')
    # newer_df = all_result_df.merge(mean_module_size, on=['method', 'deepsplit'])
    # sns.lineplot(data=newer_df, x='mean_module_size', y='nr_enriched_go_terms',
    #              hue='method', style='method', err_style='bars', marker='o')
    # plt.savefig(out_path / 'module_size_mean_nr_go_terms_line_plot.png')
    # # plt.errorbar(...)
    # plt.show()
    #
    # sns.lineplot(data=newer_df, x='mean_module_size', y='semantic_similarity',
    #              hue='method', style='method', err_style='bars', marker='o')
    # plt.savefig(out_path / 'module_size_mean_semantic_sim_line_plot.png')


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
            (all_result_df['method'] == 'local_dists')
            & (all_result_df['deepsplit'] == '2')
            | (all_result_df['method'].isin(
                ['atted_dists', 'combined_sum_dists', 'random']))
            & (all_result_df['deepsplit'] == '1')
            ]
    elif 'heat' == in_path.parent.name:
        valid_rows = all_result_df[
            (all_result_df['deepsplit'] == '1')
            & (all_result_df['method'].isin(
                ['atted_dists', 'combined_sum_dists', 'local_dists',
                 'random']))
            ]
    assert len(valid_rows) > 0
    return valid_rows


def plot_gene_modules_ds_size_distribution(all_result_df: pd.DataFrame, out_path: Path):
    size_pairs = [
        (('1', 'atted_dists'), ('1', 'combined_sum_dists')),
        (('1', 'local_dists'), ('1', 'combined_sum_dists')),
        (('1', 'atted_dists'), ('1', 'local_dists')),
        # (('2', 'atted_dists'), ('2', 'combined_sum_dists')),
        # (('2', 'local_dists'), ('2', 'combined_sum_dists')),
        # (('2', 'atted_dists'), ('2', 'local_dists')),
    ]

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

    expr_mat_time.keep_highest_z_clusters(
        hyper_params['top_nr_clusters'],
        tf2_output_path=tf2_out_path,
        plotting_path=experiment_path / 'figs')
    # expr_mat_time.plot_clusters_over_time()

    check_cutoffs = False
    if check_cutoffs:
        # Test what intermodular network looks like for various
        # correlation cutoffs
        check_correlation_cutoffs_for_intermodular_network(
            expr_mat_time,
            tf2_in_path,
            tf2_out_path,
            plotting_path=experiment_path / 'figs'
        )

    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=hyper_params['edge_corr_threshold'],
        module_plot_path=experiment_path / 'global_cluster_module_network.svg')
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
                                 ax_to_plot_on: bool | plt.Axes = False):
    """Measure coherence between different clusterings and do statistical test"""
    out_records = []
    for method in ['atted_dists', 'combined_sum_dists',
                   'local_dists', 'random']:
        for ds_filename in ['ds1']: #, 'ds2']:
            # if (method, ds_filename) == ('random', 'ds2'):
            #     continue
            if 'drought' in in_dir.parts and method == 'local_dists':
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
        ax = sns.boxplot(data=df, y='coherence', x='method')

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
                                     plot_out_path: Path | None = None):
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
        combined_distances = local_dist_flat_norm + atted_dist_flat_norm
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
