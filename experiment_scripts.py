import copy
import logging
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from random import sample
import re

import dill as pickle
import mlflow
import numpy as np
import pandas as pd
import pypesto
import pypesto.petab
import petab
import amici.petab.simulator
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
import amici
from tqdm import tqdm
from statannotations.Annotator import Annotator


from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import AggregationMethod, \
    ExpressionMatrixTimeSeries
from GoEnrich.EnrichedGeneModuleGoTerms import EnrichedGeneModuleGoTerms
from analysis_pipelines import module_network_from_tf2_output
from data_wrangling import expr_mat_from_heat, expr_mat_from_drought

from exploring_questions import plot_module_size_distributions, \
    combine_local_distance_and_prior, similarity_matrices_local_and_atted
from helpers import one_gene_list_file_per_cluster
from petab_integration.petab_scripts import write_petab_files, \
    param_optimise_petab_problem


def full_pipeline_prototype(experiment_path: Path):
    skip_slow_steps = False
    # for treatment_name in ['heat']:
    # for treatment_name in ['drought']:
    for treatment_name in ['drought', 'heat']:
        logging.info(f'Doing {treatment_name}')
        treatment_path = experiment_path / treatment_name
        data_params, hyper_params, experiment_params = config_preprocess(
            treatment_path)
        if not skip_slow_steps:
            expr_mat_all_genes = expr_mat_time_factory(
                treatment_path,
                data_params['soft_path'],
                hyper_params['agg_method'],
                hyper_params['do_log2'],
                gpl_path=data_params.get('gpl_path', None)
            )
            expr_mat_all_genes.save_for_limma(treatment_path / '01_input_for_limma.csv')

        # ## Here: run limma script (limma_de_selection/de_selection.R) ##
        # continue
        # Select only the DE genes
        de_file_path = list(treatment_path.glob('02[a_]*.csv'))
        assert len(de_file_path) == 1
        de_file_path = str(de_file_path[0])
        expr_mat_time = expr_mat_time_factory(
            treatment_path,
            de_file_path,
            hyper_params['agg_method'],
            hyper_params['do_log2'],
            gpl_path=None)
        #
        expr_mat_time.merge_biological_samples()
        # # Read DE genes from limma output and get the ATTED/Merged/Local scores
        if not skip_slow_steps:
            save_files_for_wgcna_cutting(treatment_path, data_params, expr_mat_time)
        ## Here: run wgcna cutting script (r_wgcna_dyntreecut/dyntreecut.R) ##
        # continue
        see_gene_module_sizes(expr_mat_time,
                              cut_modules_path=treatment_path / 'dyntreecut_output',
                              figure_path=treatment_path / 'figs')

        skip_making_one_file_per_clust = False
        if not skip_making_one_file_per_clust:
            one_gene_list_file_per_cluster(
                in_dir=treatment_path / 'dyntreecut_output',
                out_dir=treatment_path / 'split_by_module',
                use_for_analysis_func=lambda x: True
            )
        skip_making_random_modules = False
        # Also generate random clusters that have the same size as a representative of these clusters
        if not skip_making_random_modules:
            expr_mat_time.save_random_modules_for_goa_find_enrichment(
                wgcna_label_file=treatment_path
                                 / 'dyntreecut_output'
                                 / 'combined_sum_dists_wgcna_clustered_ds1.csv',
                out_dir=treatment_path / 'split_by_module'
            )
        # Coherence
        do_coherence_with_stat_tests(
            in_dir=treatment_path / 'split_by_module',
            expr_mat_time=expr_mat_time,
            out_dir=treatment_path / 'figs'
        )

        # continue
        # Do GO enrichment
        ### RUN SNAKEMAKE ###
        # snakemake - s.. /../../../ snakemake_workflows / Snakefile_wgcna_deepsplit_go_terms - r - c5 - k

        go_enrich_output_path = (
                treatment_path
                / 'go_outputs_exp_evidence_only_background_de_genes'
        )

        analyse_go_enrichments_find_enrichment(
            go_enrich_output_path,
            treatment_path / 'figs',
            )

        with mlflow.start_run(
                description=experiment_params['description']):
            mlflow.log_params(data_params)
            mlflow.log_params(hyper_params)
            mlflow.set_tags(experiment_params)
            mlflow.log_artifact(
                str(experiment_path / treatment_name / 'figs'))
        # ODE modelling steps


def see_gene_module_sizes(expr_mat_time: ExpressionMatrixTimeSeries,
                          cut_modules_path: Path,
                          figure_path: Path):
    out_records = []
    for dyntreecut_file in cut_modules_path.iterdir():
        expr_mat_time_copy = copy.deepcopy(expr_mat_time)
        expr_mat_time_copy.assign_clusters_from_wgcna(dyntreecut_file)
        sizes = expr_mat_time_copy.get_module_sizes()
        method, ds_value = dyntreecut_file.name.split('_wgcna_clustered_')
        ds_value = re.search('(?<=ds)\d+', ds_value).group()
        for size in sizes:
            out_records.append((size, method, ds_value))
    df = pd.DataFrame.from_records(out_records,
                                   columns=['module_size', 'method',
                                            'deepsplit'])
    plot_gene_modules_ds_size_distribution(df, figure_path)

def expr_mat_time_factory(folder: Path,
                          expression_path: str,
                          agg_method: AggregationMethod,
                          do_log2: bool,
                          gpl_path = None
                          ) -> ExpressionMatrixTimeSeries:
    if folder.name.startswith('drought'):
        expr_mat_time = expr_mat_from_drought(
            in_file_path=expression_path,
            agg_method=agg_method,
            do_log2=do_log2)
    elif folder.name.startswith('heat'):
        expr_mat_time = expr_mat_from_heat(in_path=expression_path,
                                           agg_method=agg_method,
                                           do_log2=do_log2,
                                           gpl_path=gpl_path)
    else:
        raise NotImplementedError
    return expr_mat_time


def module_size_pipeline(experiment_path):
    for file in experiment_path.iterdir():
        if file.name.endswith('expr_mat_dict.pkl'):
            plot_module_size_distributions(file)
    with mlflow.start_run():
        for file in experiment_path.iterdir():
            mlflow.log_artifact(str(file))
            # if not file.suffix in ['.npy', '.pkl', '.gzip']:
            #     mlflow.log_artifact(str(file))

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



def save_jackknife_files(experiment_path, expr_mat_time, condition_name):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)

    (atted_dist_df, atted_score,
     condition_base_path, local_dist,
     min_dist_df, sum_dist_df) = save_files_for_wgcna_cutting(experiment_path,
                                                              data_params,
                                                              expr_mat_time)

    generate_jackknifes(atted_dist_df, atted_score, condition_base_path,
                        hyper_params, local_dist, min_dist_df, sum_dist_df)


def generate_jackknifes(atted_dist_df, atted_score, condition_base_path,
                        hyper_params, local_dist, min_dist_df, sum_dist_df):
    out_records = []
    do_debug_thing = False
    # Save jackknifed distance matrices
    for i in range(hyper_params['nr_jackknifes']):
        logging.info(f'Iteration {i + 1}')
        rand_index = sample(
            local_dist.index.tolist(),
            round(hyper_params['subset_size'] * len(local_dist.index.tolist()))
        )
        # From subset get local and global dists
        subset_local_df = local_dist.loc[rand_index, rand_index]
        subset_atted_df = atted_score.loc[rand_index, rand_index]

        local_jackknife_dir = condition_base_path / 'jackknifes' / 'local'
        local_jackknife_dir.mkdir(exist_ok=True)
        subset_local_df.to_parquet(local_jackknife_dir
                                   / f'jackknife_{i}.parquet.gzip',
                                   compression='gzip')

        atted_jackknife_dir = condition_base_path / 'jackknifes' / 'atted'
        atted_jackknife_dir.mkdir(exist_ok=True)
        subset_atted_dist_df = subset_atted_df.max().max() - subset_atted_df
        subset_atted_dist_df.to_parquet(atted_jackknife_dir
                                        / f'jackknife_{i}.parquet.gzip',
                                        compression='gzip')

        min_jackknife_dir = condition_base_path / 'jackknifes' / 'combined_min'
        min_jackknife_dir.mkdir(exist_ok=True)
        # And min dists
        subset_min_dist_df = combine_local_distance_and_prior(
            subset_local_df,
            subset_atted_df,
            dists_out_path=(min_jackknife_dir
                            / f'jackknife_{i}.parquet.gzip'),
            combo='min',
            calculate_linkages=False,
            plot_out_path=None,
        )

        # And sum dists
        sum_jackknife_dir = condition_base_path / 'jackknifes' / 'combined_sum'
        sum_jackknife_dir.mkdir(exist_ok=True)
        subset_sum_dist_df = combine_local_distance_and_prior(
            subset_local_df,
            subset_atted_df,
            dists_out_path=(sum_jackknife_dir / f'jackknife_{i}.parquet.gzip'),
            combo='sum',
            calculate_linkages=False,
            plot_out_path=None,
        )

        if do_debug_thing:
            # Just for one jackknife atm
            def debug_func(dist1, dist2, nclust=50):
                a = cluster_from_dists(dist1, nclust)
                b = cluster_from_dists(dist2, nclust)

                merged_df = a.join(b, how='inner',
                                   lsuffix='_subset',
                                   rsuffix='_og')

                return adjusted_rand_score(merged_df['colors_subset'],
                                           merged_df['colors_og'])

            def cluster_from_dists(dist, nclust=50):
                clustering = fcluster(
                    linkage(squareform(dist, checks=False), method='average'),
                    nclust, 'maxclust')
                return pd.DataFrame(clustering, index=dist.index,
                                    columns=['colors'])

            for nclust in range(1, 500, 50):
                out_records.append(
                    ('local', debug_func(
                        subset_local_df, local_dist, nclust=nclust),
                     nclust))
                out_records.append(
                    ('min_dist', debug_func(
                        min_dist_df, subset_min_dist_df, nclust=nclust),
                     nclust))
                out_records.append(
                    ('sum_dist', debug_func(
                        sum_dist_df, subset_sum_dist_df, nclust=nclust),
                     nclust))
                out_records.append(
                    ('atted', debug_func(
                        subset_atted_dist_df, atted_dist_df, nclust=nclust),
                     nclust))
    if do_debug_thing:
        plot_df = pd.DataFrame.from_records(out_records,
                                            columns=['dist', 'ari', 'nclust'])
        sns.lineplot(data=plot_df, x='nclust', y='ari', hue='dist')
        plt.show()


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
    sum_dist_df = combine_local_distance_and_prior(
        local_dist,
        atted_score,
        dists_out_path=(experiment_path
                        / 'full_datasets' / 'combined_sum_dists.parquet.gzip'),
        combo='sum',
        calculate_linkages=False,
        plot_out_path=fig_folder,
    )

    atted_dist_df = atted_score.max().max() - atted_score
    atted_dist_df.to_parquet(experiment_path
                             / 'full_datasets' / 'atted_dists.parquet.gzip',
                             compression='gzip')
    return atted_dist_df, atted_score, experiment_path, local_dist, sum_dist_df


def wgcna_with_similarity_scores(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    expr_mat_time = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])
    similarity_matrices_local_and_atted(expr_mat_time, data_params['atted_path'],
                                        out_path=experiment_path)

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

def analyse_go_enrichments_find_enrichment(in_path: Path, out_path: Path):
    # For DS and Method
    all_result_df = read_go_enrich_files_into_df(in_path)
    # plot_gene_modules_ds_size_distribution(all_result_df, out_path)
    # valid_rows = extract_only_selected_ds_row_from_df(all_result_df, in_path)
    valid_rows = all_result_df
    # Main figures
    # Fraction of modules with > 0 GO term
    at_least_one_go_term_barplot_keywords = dict(
        data=valid_rows, y='nr_enriched_go_terms', x='method',
        estimator=lambda y: (y > 0).sum() / len(y)
    )
    plt.close()
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
    # plt.tight_layout()
    plt.savefig(out_path / 'fraction_at_least_one_go_term_selected_ds.svg',
                bbox_inches='tight')
    plt.close()

    # GO semantic similarity scores
    ax =  sns.boxplot(data=valid_rows, y='semantic_similarity', x='method')
    annotator.new_plot(ax, pairs, data=valid_rows, y='semantic_similarity', x='method')
    annotator.apply_and_annotate()
    plt.savefig(out_path / 'semantic_similarity_boxplot.svg', bbox_inches='tight')
    plt.close()

    # Other figures
    ax = sns.boxplot(data=valid_rows, y='nr_enriched_go_terms', x='method')
    annotator.new_plot(ax, pairs, data=valid_rows, y='nr_enriched_go_terms',
                          x='method')
    annotator.apply_and_annotate()
    plt.savefig(out_path / 'go_terms_per_module_boxplot_selected_ds.svg', bbox_inches='tight')
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


def config_preprocess(experiment_path) -> tuple[dict, dict, dict]:
    config_path = experiment_path / 'config.yaml'
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    data_params = config['data']
    hyper_params = config['hyperparams']
    experiment_params = config['experiment_data']
    agg_method_dict = {'mean': AggregationMethod.MEAN,
                       'eigengene': AggregationMethod.EIGENGENE}
    hyper_params['agg_method'] = agg_method_dict.get(
        hyper_params.get('agg_method')
    )
    return data_params, hyper_params, experiment_params

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


def from_expr_mat_time_to_ode(data_params, experiment_path, expr_mat_time,
                              hyper_params):
    wgcna_module_assignment = data_params['wgcna_module_assignment_path']
    expr_mat_time.assign_clusters_from_wgcna(wgcna_module_assignment)
    skip_stuff = True
    if not skip_stuff:
        atted_score = pd.read_parquet(data_params['atted_path'])
        atted_score = atted_score.set_index(atted_score.columns[0])
        linkage_matrices = combine_local_distance_and_prior(
            expr_mat_time.get_distance_matrix(),
            atted_score,
            out_path=experiment_path)
    # logging.warning('New clusters so new TF2Network analysis?')
    tf2_in_path = experiment_path / data_params['tf2_in_name']
    tf2_out_path = experiment_path / data_params['tf2_out_name']
    # Post to tf2network
    expr_mat_time.write_tf2_input_file(
        out_path=tf2_in_path)
    # expr_mat_time.do_genewise_normalisation()
    expr_mat_time.keep_highest_z_clusters(
        hyper_params['top_nr_clusters'],
        tf2_output_path=tf2_out_path,
        plotting_path=experiment_path)
    # expr_mat_time.plot_clusters_over_time()
    expr_mat_time.get_ci_per_cluster()
    # # TO get gene list
    # [print(i) for i in expr_mat_time.get_genes_per_cluster()[75]]
    expr_mat_time.do_genewise_min_max_scaling()
    expr_mat_time.get_ci_per_cluster()
    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=hyper_params['edge_corr_threshold'],
        module_plot_path=experiment_path / 'global_cluster_module_network.svg')
    expr_mat_time.keep_only_modules_in_network(module_module)
    expr_mat_time.plot_clusters_over_time()
    with (experiment_path / 'expr_mat_time.pkl').open('wb') as f:
        pickle.dump(expr_mat_time, f)
    with (experiment_path / 'module_network.pkl').open('wb') as f:
        pickle.dump(module_module, f)
    # Assure that data has already been clustered
    assert expr_mat_time.has_been_clustered
    # expr_mat_time.get_genes_per_cluster()[328]
    my_ode = OdeModel.construct_from_regulatory_network(module_module,
                                                        nonlinear=True)
    return my_ode

def ground_truth_vs_jackknife(experiment_path, expr_mat_time):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)

    in_path = Path(data_params['r_out_path'])
    out_list = []
    module_size_list = []
    # FOr keyword, find jackknifes:
    for method in ['atted', 'combined_min', 'combined_sum', 'local']:
        for deepsplit_value in hyper_params['r_deep_split']:
            expr_mat_time_copy = deepcopy(expr_mat_time)
            jackknife_paths = list(
                in_path.glob(f'{method}/*ds{deepsplit_value}.csv')
            )
            logging.info(f'Doing {method} with {deepsplit_value=}')
            # Full dataset
            full_dataset_path_list = list(
                in_path.glob(f'full_datasets/{method}*ds{deepsplit_value}.csv')
            )
            assert len(jackknife_paths) == hyper_params['nr_jackknifes']
            assert  len(full_dataset_path_list) == 1
            full_dataset_path = full_dataset_path_list[0]
            full_dataset = pd.read_csv(full_dataset_path, usecols=[1,2])
            full_dataset = full_dataset.set_index('gene_id')

            # Module sizes (in full dataset)
            module_size_list.extend(
                [(method, deepsplit_value, size)
                for size in full_dataset.value_counts().to_list()]
            )
            # sns.histplot(full_dataset.value_counts().to_list())
            # plt.xlabel('Module size')
            # plt.savefig(in_path.parent / 'figs'
            #             / f'{method}_size_modules_ds{deepsplit_value}.png')
            # plt.close()

            # Do coherence per module
            expr_mat_time_copy.assign_clusters_from_wgcna(full_dataset_path)
            coherence_entry = expr_mat_time_copy.get_all_explained_vars()
            out_list.extend([(method, deepsplit_value, 'coherence', i) for i in coherence_entry])

            # Robustness
            for jackknife_path in jackknife_paths:
                    subset = pd.read_csv(jackknife_path, usecols=[1,2])
                    subset = subset.set_index('gene_id')
                    # Merge genes
                    merged_df = subset.join(full_dataset,
                                            how='inner',
                                                       lsuffix='_subset',
                                                       rsuffix='_og')
                    ari = adjusted_rand_score(merged_df['colors_subset'],
                                              merged_df['colors_og'])
                    out_list.append((method, deepsplit_value, 'robustness', ari))

    metric_df = pd.DataFrame.from_records(out_list, columns=['method', 'deepsplit', 'metric', 'score'])

    module_size_df = pd.DataFrame.from_records(module_size_list, columns=['method', 'deepsplit', 'size'] )
    module_size_df['deepsplit'] = module_size_df['deepsplit'].astype('str')
    module_size_df['size'] = module_size_df['size'].astype('int')

    sns.boxplot(data=module_size_df, x='size', y='deepsplit', hue='method')
    plt.savefig(in_path.parent /  'figs' / 'module_size_hue_is_ds.png')

    sns.catplot(data=module_size_df, x='deepsplit', col='method',
                col_wrap=2, kind='count')
    plt.savefig(in_path.parent / 'figs' / 'module_counts.png')
    plt.close()

    # Select deepsplit value on most comparable module sizes
    if 'drought' in data_params['in_path']:
        valid_rows = metric_df[
            (metric_df['method'] == 'local') & (metric_df['deepsplit'] == 2)
            | (metric_df['method'].isin(['atted', 'combined_sum'])) & (metric_df['deepsplit'] == 1) ]
    elif 'heat' in data_params['in_path']:
        valid_rows = metric_df[
            (metric_df['deepsplit'] == 1) & (metric_df['method'].isin(['atted', 'combined_sum', 'local']))]

    sns.catplot(valid_rows, x='method', y='score', row='metric',
                kind='box', hue='method', )
    plt.savefig(
        in_path.parent / 'figs' / 'coherence_robustness_modules_selected_ds_separate_rows_boxplot.png')

    sns.catplot(valid_rows, x='method', y='score', row='metric',
                kind='violin', hue='method')
    plt.savefig(
        in_path.parent / 'figs' / 'coherence_robustness_modules_selected_ds_separate_rows_violin.png')

    sns.catplot(metric_df, x='deepsplit', y='score', hue='metric',
                col='method', kind='box')
    plt.savefig(in_path.parent /  'figs' / 'coherence_robustness_modules.png')
    plt.close()



    # sns.violinplot(data=robustness_df, y='ari', x='method')
    # plt.savefig(in_path.parent /  'figs' / 'robustness_modules.png')
    # plt.close()
    #
    # sns.violinplot(data=coherence_df, y='coherence', x='method')
    # plt.savefig(in_path.parent / 'figs' / 'coherence_modules.png')
    # plt.close()


def pypesto_from_sbml(experiment_path, condition):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path
    )
    with open(data_params['expr_mat_time_path'], 'rb') as f:
        expr_mat_time: ExpressionMatrixTimeSeries = pickle.load(f)

    if hyper_params['do_gene_normalisation']:
        expr_mat_time.do_genewise_min_max_scaling()

    write_petab_files(
        expr_mat_time,
        data_params['sbml_path'],
        experiment_path / 'petab_files',
        experimental_setup=condition
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
                                 out_dir: Path):
    """Measure coherence between different clusterings and do statistical test"""
    out_records = []
    for method in ['atted_dists', 'combined_sum_dists',
                   'local_dists', 'random']:
        for ds_filename in ['ds1']: #, 'ds2']:
            # if (method, ds_filename) == ('random', 'ds2'):
            #     continue
            expr_mat_time_copy = copy.deepcopy(expr_mat_time)
            pattern = f"{method}*{ds_filename}*"
            files = list(in_dir.glob(pattern))
            assert len(files) > 0, f'No files found for {method} & {ds_filename}'
            expr_mat_time_copy.assign_clusters_from_split_by_module_files(files)
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

    out_dir.mkdir(exist_ok=True)
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
    plt.ylim((0, .9))

    plt.savefig(out_dir / 'boxplot_coherence_with_stat_test.svg',
                bbox_inches='tight')
    plt.close()
