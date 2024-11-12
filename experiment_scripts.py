import logging
from copy import deepcopy
from pathlib import Path
from random import sample
from typing import Dict
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

from DynamicModels.OdeFitterMultipleDatasets import OdeFitterMultipleDatasets
from DynamicModels.OdeLocalParameters import OdeLocalParameters

from DynamicModels.OdeModel import OdeModel
from DynamicModels.helper_scripts_for_fitting import fit_multiple_fitters
from Expressions.ExpressionMatrix import AggregationMethod, \
    ExpressionMatrixTimeSeries
from GoEnrich.EnrichedGeneModuleGoTerms import EnrichedGeneModuleGoTerms
from analysis_pipelines import explore_emtab_375, compare_clusterings_for_ode_use, \
    module_network_from_tf2_output, infer_intermodular_network
from data_wrangling import expr_mat_from_emexp, expr_mat_from_drought

from exploring_questions import plot_module_size_distributions, \
    combine_local_distance_and_prior, similarity_matrices_local_and_atted
from helpers import parse_string_input_data
from petab_integration.petab_scripts import write_petab_files_heat, \
    param_optimise_petab_problem


def prefilter_genes_experiment(experiment_path):

    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)

    expr_mat_time_drought = expr_mat_from_drought(
        data_params['limma_drought_out_path'],
        hyper_params['agg_method'],
        hyper_params['do_log2_drought'])

    expr_mat_time_heat = expr_mat_from_emexp(
        data_params['limma_heat_out_path'],
        hyper_params['agg_method'],
        hyper_params['do_log2_heat'],
        data_params['heat_gpl_path']
    )

    cv_list = []
    df_list = []


    expr_mat_time_heat
    #
    #
    # for expr_mat_time, condition_name in zip(
    #         [expr_mat_time_drought, expr_mat_time_heat],
    #         ['drought', 'heat']
    # ):
    #     expr_mat_time.scatterplot_of_two_per_gene_stats(
    #         'std', 'cond_rmsd',
    #         plotting_func=sns.jointplot,
    #         title = f'{condition_name} _std_rmsd_no cutoff ({len(expr_mat_time.df)} genes)',
    #         out_path=experiment_path /  f'{condition_name}_no_cutoff.png')
    #
    #     # expr_mat_time.scatterplot_of_two_per_gene_stats(
    #     #     'mean', 'std',
    #     #     plotting_func=sns.jointplot,
    #     #     title = f'{condition_name} no cutoff ({len(expr_mat_time.df)} genes)',
    #     #     out_path=experiment_path /  f'{condition_name}_no_cutoff.png')
    #
    #     for cutoff in [0.25, 0.5, 0.75]:
    #         temp_expr_mat = deepcopy(expr_mat_time)
    #         # std_series = temp_expr_mat.plot_per_gene_std()
    #         temp_expr_mat.keep_genes_above_percentile_score(
    #             cutoff,
    #             method='cond_rmsd')
    #         # temp_expr_mat.scatterplot_of_two_per_gene_stats(
    #         #     'mean', 'std',
    #         #     plotting_func=sns.jointplot,
    #         #     title=f'{condition_name} cutoff={cutoff} perc. ({len(temp_expr_mat.df)} genes)',
    #         #     out_path=experiment_path / f'{condition_name}_{cutoff}_cutoff.png'
    #         # )
    #         # mad_series = expr_mat_time.plot_per_gene_mad()
    #         # cv_serie = expr_mat_time._calculate_gene_variation('cv')
    #         # cv_list.append(cv_serie)
    #         #
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('std', 'qcd')
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('mean', 'qcd')
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('std', 'mad')
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('std', 'cv')
    #         # expr_mat_time.scatterplot_of_two_per_gene_stats('std', 'qcd')
    #         # Median should roughly be good cutoff?
    #         # Or only remove lower 25th percentile?
    #         # Look at distribution of MAD
    #
    #
    # # cv_list[0].name = 'drought'
    # # cv_list[1].name = 'heat'
    # # merged_df = pd.concat(cv_list, axis=1, join='inner')
    # # sns.histplot(merged_df)
    # # print()






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
                in_file_path=data_params['soft_path'],
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
    expr_mat_time: ExpressionMatrixTimeSeries = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
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

def integrate_multiple_datasets(experiment_path):
    # Download some GEO here
    expr_mat_time_supp = ExpressionMatrixTimeSeries.from_xlsx(
        'data/raw_data/expression_datasets/GSE134945/GSE134945_readcount.xlsx')
    expr_mat_time_supp.keep_n_most_deviating_genes(2000, plot=True)
    # TODO THINK ABOUT TPKM NORMALISE OR SMTH?
    # sns.clustermap(expr_mat_time_supp.get_correlation_matrix())
    # plt.show()

    # data_params, hyper_params, experiment_params = config_preprocess(
    #     experiment_path)
    expr_mat_time_og = expr_mat_from_drought(
        'limma_de_selection/drought_expr_matrix_limma_filtered.csv',
                      'mean',
                      False)

    linkage_matrices = combine_local_distance_and_prior(
        expr_mat_time_og.get_distance_matrix(absolute_dist=False),
        expr_mat_time_supp.get_distance_matrix(absolute_dist=False),
        out_path=experiment_path,
        combo='sum')



def generate_dists_for_wgcna_cutting(experiment_path, expr_mat_time, condition_name):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)

    condition_base_path = experiment_path / condition_name

    for folder_name in ['figs', 'full_datasets', 'jackknifes']:
        new_folder = condition_base_path / folder_name
        new_folder.mkdir(parents=True, exist_ok=True)
    # expr_mat_time = expr_mat_from_emexp(data_params['in_path'],
    #                                       hyper_params['agg_method'],
    #                                       hyper_params['do_log2'])
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
    local_dist.to_parquet(condition_base_path
                          / 'full_datasets' / 'local_dists.parquet.gzip',
                               compression='gzip')

    # And combined dists
    min_dist_df = combine_local_distance_and_prior(
        local_dist,
        atted_score,
        dists_out_path=(condition_base_path
                        / 'full_datasets' / 'combined_min_dists.parquet.gzip'),
        combo='min',
        calculate_linkages=False,
        plot_out_path=None,
    )

    # And combined dists
    sum_dist_df = combine_local_distance_and_prior(
        local_dist,
        atted_score,
        dists_out_path=(condition_base_path
                        / 'full_datasets' / 'combined_sum_dists.parquet.gzip'),
        combo='sum',
        calculate_linkages=False,
        plot_out_path=None,
    )

    atted_dist_df = atted_score.max().max() - atted_score
    atted_dist_df.to_parquet(condition_base_path
                             / 'full_datasets' / 'atted_dists.parquet.gzip',
                                   compression='gzip')

    out_records = []
    do_debug_thing = False
    # Save jackknifed distance matrices
    for i in range(hyper_params['nr_jackknifes']):
        logging.info(f'Iteration {i+1}')
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


            for nclust in range(1,500, 50):
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


def wgcna_with_similarity_scores(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    expr_mat_time = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])
    similarity_matrices_local_and_atted(expr_mat_time, data_params['atted_path'],
                                        out_path=experiment_path)

def drought_data_e2e_pipeline(experiment_path):
    # Load the config file
    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)
    expr_mat_time = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])
    abs_dists = False
    skip_stuff = True
    if not skip_stuff:
        atted_score = pd.read_parquet(data_params['atted_path'])
        atted_score = atted_score.set_index(atted_score.columns[0])
        linkage_matrices = combine_local_distance_and_prior(
            expr_mat_time.get_distance_matrix(absolute_dist=abs_dists),
            atted_score,
            combo='min',
            out_path=experiment_path
            )

    expr_mat_time.assign_clusters_from_wgcna(data_params['wgcna_cluster_path'])

    expr_mat_time, module_module = infer_intermodular_network(
        expr_mat_time=expr_mat_time,
        experiment_path=experiment_path,
        tf2_in_name=data_params['tf2_in_name'],
        tf2_out_name=data_params['tf2_out_name'],
        top_nr_clusters=hyper_params['top_nr_clusters'],
        edge_cor_threshold=hyper_params['edge_corr_threshold']
    )


    with (experiment_path / 'module_network.pkl').open('wb') as f:
        pickle.dump(module_module, f)
    # Assure that data has already been clustered
    assert expr_mat_time.has_been_clustered
    # expr_mat_time.get_genes_per_cluster()[328]
    fit_ode_drought_data(experiment_path, expr_mat_time, hyper_params,
                         module_module)


def analyse_go_enrichments_find_enrichment(in_path: Path, out_path: Path):
    # For DS and Method
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
            nr_enriched= 0
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

    # sns.scatterplot(data=all_result_df, x='nr_enriched_go_terms',
    #                 y='semantic_similarity', hue='method')
    # plt.show()

    sns.scatterplot(data=all_result_df, x='module_size',
                    y='nr_enriched_go_terms', hue='method', style='deepsplit')
    plt.savefig(out_path / 'go_terms_module_scatterplot.png')
    plt.close()

    sns.boxenplot(data=all_result_df, x='deepsplit', y='nr_enriched_go_terms',
                  hue='method')
    plt.savefig(out_path / 'go_terms_ds_boxenplot.png')
    plt.close()


    # print(all_result_df.groupby(['method', 'deepsplit'])['nr_enriched_go_terms'].mean())
    sns.catplot(data=all_result_df, x='deepsplit', y='nr_enriched_go_terms',
                col='method', kind='box')
    plt.savefig(out_path / 'go_terms_per_module_boxplot.png')
    plt.close()

    sns.boxplot(data=all_result_df, x='module_size', y='method',
                hue='deepsplit')
    plt.tight_layout()
    plt.savefig(out_path / 'module_sizes_deepsplit_hue_is_deepsplit_boxplot.png')
    plt.show()

    sns.boxplot(data=all_result_df, x='module_size', hue='method',
                y='deepsplit')
    plt.tight_layout()
    plt.savefig(out_path / 'module_sizes_deepsplit_hue_is_method_boxplot.png')
    plt.show()

    sns.catplot(data=all_result_df, x='deepsplit', y='nr_enriched_go_terms',
                col='method', kind='strip')
    plt.savefig(out_path / 'go_terms_per_module_stripplot.png')
    plt.close()

    # Select deepsplit value on most comparable module sizes
    if np.all(all_result_df['method'] == 'random'):
        valid_rows = all_result_df
    elif 'drought' == in_path.parent.name:
        valid_rows = all_result_df[
            (all_result_df['method'] == 'local_dists')
            & (all_result_df['deepsplit'] == '2')
            | (all_result_df['method'].isin(
                ['atted_dists', 'combined_sum_dists']))
            & (all_result_df['deepsplit'] == '1')
            ]
    elif 'heat' == in_path.parent.name:
        valid_rows = all_result_df[
            (all_result_df['deepsplit'] == '1')
            & (all_result_df['method'].isin(
                ['atted_dists', 'combined_sum_dists', 'local_dists']))
            ]
    assert len(valid_rows) > 0
    sns.boxplot(data=valid_rows, y='nr_enriched_go_terms', x='method',
                hue='method')
    plt.tight_layout()
    plt.savefig(out_path / 'go_terms_per_module_boxplot_selected_ds.png')
    plt.close()

    # Fraction of modules with > 0 GO term
    ax = sns.barplot(data=valid_rows, y='nr_enriched_go_terms', x='method',
                hue='method', estimator=lambda x: (x != 0).sum() / len(x))
    ax.set_ylabel('Fraction of modules with > 0 enriched GO term')
    plt.savefig(out_path / 'fraction_at_least_one_go_term_selected_ds.png')
    plt.close()

    # Of these GO-terms, what is their semantic similarity?
    sns.scatterplot(data=valid_rows, x='nr_enriched_go_terms',
                    y='semantic_similarity', hue='method')
    plt.savefig(out_path / 'scatterplot_enriched_go_terms_semantic_sim_selected_ds.png')
    plt.close()

    sns.jointplot(data=valid_rows, x='nr_enriched_go_terms',
                    y='semantic_similarity', hue='method')
    plt.savefig(out_path / 'jointplot_enriched_go_terms_semantic_sim_selected_ds.png')
    plt.close()

    sns.barplot(data=valid_rows, y='semantic_similarity', x='method')
    plt.savefig(out_path / 'semantic_similarity_barplot.png')
    plt.close()

    sns.boxplot(data=valid_rows, y='semantic_similarity', x='method')
    plt.savefig(out_path / 'semantic_similarity_boxplot.png')
    plt.close()

    mean_module_size = all_result_df.groupby(
        ['method', 'deepsplit'])['module_size'].mean()
    mean_enriched_go_terms = all_result_df.groupby(
        ['method', 'deepsplit'])['nr_enriched_go_terms'].mean()

    mean_semantic_similarity = all_result_df.groupby(
        ['method', 'deepsplit'])['semantic_similarity'].mean()


    mean_module_size_and_go_enrichments = pd.concat(
        [mean_module_size, mean_enriched_go_terms, mean_semantic_similarity], axis=1).reset_index()
    mean_module_size_and_go_enrichments = mean_module_size_and_go_enrichments.rename(
        {'module_size': 'mean_module_size',
            'semantic_similarity': 'mean_semantic_similarity'}, axis='columns')

    sns.scatterplot(data=mean_module_size_and_go_enrichments, x='mean_module_size',
                    y='nr_enriched_go_terms', hue='method', style='deepsplit')
    plt.ylabel('Mean nr of enriched go terms per module')
    # Add error bar?
    plt.savefig(out_path / 'module_size_mean_nr_go_terms_scatterplot.png')
    plt.show()
    plt.close()

    sns.scatterplot(data=mean_module_size_and_go_enrichments, x='mean_module_size',
                    y='mean_semantic_similarity', hue='method', style='deepsplit')
    plt.ylabel('Mean semantic similarity within a module')
    plt.savefig(out_path / 'module_size_mean_semantic_sim_scatterplot.png')
    # Add error bar?
    plt.show()
    plt.close()

    # sns.scatterplot(data=all_result_df, x='module_size',
    #                 y='semantic_similarity', hue='method', style='deepsplit')
    # plt.show()


    mean_module_size = mean_module_size.reset_index()
    mean_module_size = mean_module_size.rename(
        {'module_size': 'mean_module_size'}, axis='columns')
    newer_df = all_result_df.merge(mean_module_size, on=['method', 'deepsplit'])
    sns.lineplot(data=newer_df, x='mean_module_size', y='nr_enriched_go_terms',
                 hue='method', style='method', err_style='bars', marker='o')
    plt.savefig(out_path / 'module_size_mean_nr_go_terms_line_plot.png')
    # TODO implement error bars on X axis
    # plt.errorbar(...)
    plt.show()

    sns.lineplot(data=newer_df, x='mean_module_size', y='semantic_similarity',
                 hue='method', style='method', err_style='bars', marker='o')
    plt.savefig(out_path / 'module_size_mean_semantic_sim_line_plot.png')
    print()


def fit_ode_drought_data(experiment_path, expr_mat_time, hyper_params,
                         module_module):
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
        experiment_path=experiment_path,
        param_limit=hyper_params.get('param_limit')
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
    hyper_params['agg_method'] = agg_method_dict.get(
        hyper_params.get('agg_method')
    )
    return data_params, hyper_params, experiment_params

def drought_with_string_db(experiment_path):
    # Load the config file
    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)
    expr_mat_time = expr_mat_from_drought(data_params['in_path'],
                                          hyper_params['agg_method'],
                                          hyper_params['do_log2'])
    abs_dists = False
    skip_stuff = False
    # expr_mat_time.do_genewise_normalisation()
    expr_mat_time.keep_n_most_deviating_genes(2000)
    if not skip_stuff:
        prior_score = parse_string_input_data()
        linkage_matrices = combine_local_distance_and_prior(
            expr_mat_time.get_distance_matrix(absolute_dist=abs_dists),
            prior_score,
            out_path=experiment_path,
            combo='sum')
    print()


def exploratory_heat_data_scripts(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(experiment_path)
    explore_emtab_375(experiment_path=experiment_path,
                      in_file_path=Path(data_params['soft_path']),
                      summed_linkage_matrix=data_params['linkage_path'],
                      summed_dist_matrix_path=Path(
                          data_params['dist_matrix_path']),
                      nr_clusters=hyper_params['nr_clusters'],
                      do_log2=hyper_params['do_log2'],
                      agg_method=hyper_params['agg_method'],
                      gpl_path=data_params['gpl_path'])


def heat_data_e2e_pipeline(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path)
    expr_mat_time = expr_mat_from_emexp(data_params['in_path'],
                                        hyper_params['agg_method'],
                                        hyper_params['do_log2'],
                                        )
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

    expr_mat_time.plot_clusters_over_time()
    # # TO get gene list
    # [print(i) for i in expr_mat_time.get_genes_per_cluster()[75]]

    with (experiment_path / 'expr_mat_time.pkl').open('wb') as f:
        pickle.dump(expr_mat_time, f)

    module_module = module_network_from_tf2_output(
        expr_mat_time, tf2_in_path,
        tf2_out_path,
        threshold=hyper_params['edge_corr_threshold'],
        module_plot_path=experiment_path / 'global_cluster_module_network.svg')

    expr_mat_time.keep_only_modules_in_network(module_module)

    # expr_mat_time, module_module =  assign_clusters_and_infer_intermodular_network(
    #     experiment_path=experiment_path,
    #     expr_mat_time=expr_mat_time,
    #     summed_linkage_matrix=data_params['linkage_path'],
    #     summed_dist_matrix_path=Path(data_params['dist_matrix_path']),
    #     nr_clusters=hyper_params['nr_clusters'],
    #     edge_cor_threshold=hyper_params['edge_corr_threshold'],
    #     top_nr_clusters=hyper_params['top_nr_clusters'],
    #     tf2_in_name=data_params['tf2_in_name'],
    #     tf2_out_name=data_params['tf2_out_name'])

    with (experiment_path / 'module_network.pkl').open('wb') as f:
        pickle.dump(module_module, f)
    # Assure that data has already been clustered
    assert expr_mat_time.has_been_clustered
    # expr_mat_time.get_genes_per_cluster()[328]
    my_ode = OdeModel.construct_from_regulatory_network(module_module,
                                                        nonlinear=True)

    # # These are parameters that are different between the two datasets
    u_t_function = 'temp'


    # custom_params[control_name] = OdeLocalParameters(
    #     u_t=(lambda t: 0))
    # custom_params[treatment_name] = OdeLocalParameters(
    #     u_t=(lambda t: 1))

    # # Expression that determine the value of u(t)
    # custom_params[control_name] = '0 + time * 0'
    # custom_params[treatment_name] = '10 + time * 10'

    # # Expression that determine the value of u(t)
    # custom_params[control_name] = 'time'
    # custom_params[treatment_name] = 'time'

    my_ode.save_to_sbml(experiment_path / 'module_network.xml', u_t_function)
    return

    best_ode_fit = fit_ode_to_two_datasets(
        my_ode,
        expr_mat_time,
        custom_params=custom_params,
        nr_ode_iters=hyper_params['nr_ode_iters'],
        experiment_path=experiment_path,
        param_limit=hyper_params.get('param_limit'),
        gradient_matching=hyper_params['do_gradient_matching'],
        nr_fitters=hyper_params['nr_fitters'],
        nr_time_points_interpolation=hyper_params['nr_time_points_interpolation']
    )
    with (experiment_path / 'pickled_ode_model.pkl').open('wb') as f:
        pickle.dump(best_ode_fit, f)
    return expr_mat_time, module_module

def fit_ode_to_two_datasets(
        my_ode: OdeModel,
        my_time_series_expressions: ExpressionMatrixTimeSeries,
        nr_ode_iters: int,
        custom_params: Dict,
        nr_fitters: int = 5,
        experiment_path: Path|None = None,
        param_limit: float = .1,
        gradient_matching: bool = False,
        nr_time_points_interpolation = None
        ):

    # condition_names = list(custom_params.keys())
    # # Step uno
    # my_fitter = OdeFitterMultipleDatasets(my_ode,
    #                                       my_time_series_expressions,
    #                                       custom_params,
    #                                       param_limit=param_limit)
    # my_fitter.fit(max_iter=400)
    # my_fitter.calculate_current_best_fits()

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

    multiple_fitters = [
        OdeFitterMultipleDatasets(
            my_ode, my_time_series_expressions,
            custom_params,
            param_limit=param_limit,
            do_spline_smooth=gradient_matching,
            nr_time_points_interpolation=nr_time_points_interpolation
        ) for _ in range(nr_fitters)]

    best_fit = fit_multiple_fitters(multiple_fitters, nr_ode_iters)
    best_fit.calculate_current_best_fits(data_point_overlay=True,
                                         use_err_bars=True,
                                         out_path=experiment_path / 'final_ode_fit.svg')
    return best_fit
    # multiple_fitter.fit(100)
    # best_fits = multiple_fitter.calculate_current_best_fits()


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


def heat_pypesto(experiment_path):
    data_params, hyper_params, experiment_params = config_preprocess(
        experiment_path
    )

    with open(data_params['expr_mat_time_path'], 'rb') as f:
        expr_mat_time = pickle.load(f)
    model_name = "model_heat"
    model_dir = str(experiment_path / "model_dir")
    constant_parameters = ['temp']

    write_petab_files_heat(
        expr_mat_time,
        data_params['sbml_path'],
        experiment_path / 'petab_files'
    )
    # Experimental data
    petab_problem = petab.v1.Problem.from_yaml(
        str(experiment_path / 'petab_files' / 'baddadan_heat.yaml')
    )
    only_sim = False
    if not only_sim:
        result_og = param_optimise_petab_problem(petab_problem)

    importer = pypesto.petab.PetabImporter(petab_problem,
                                           simulator_type="amici",
                                           )
    factory = importer.create_objective_creator()
    obj = factory.create_objective()

    # SIMULATED DATAAAAAAAAAAA
    petab_problem_synthetic = petab.v1.Problem.from_yaml(
        str(experiment_path / 'petab_files' / 'baddadan_heat.yaml')
    )

    simulation_param_dict = {}
    for param_name in petab_problem_synthetic.parameter_df.index:
        if param_name == 'delta_0':
            value = -1
        elif 'delta' in param_name:
            value = -.1
        elif param_name == 'gamma_0':
            value = 3
        elif 'gamma' in param_name:
            value = -.1
        elif 'beta_0_1' in param_name:
            value = .01
        elif 'k_1_2' == param_name:
            value = 6
        elif 'beta' in param_name:
            value = 1
        elif 'k_0_1' == param_name:
            value = 1
        elif param_name.startswith('k_'):
            value = 2
        else:
            raise NotImplementedError
        simulation_param_dict[param_name] = value
    obj.amici_model.setInitialStates([3, 3, 3])
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

    result_sim = param_optimise_petab_problem(petab_problem_synthetic)








    # # Everything below is amici-specific and not needed at the moment
    #
    # omit_sbml_converstion = False
    # if not omit_sbml_converstion:
    #     sbml_importer = amici.SbmlImporter(data_params['sbml_path'],
    #                                        show_sbml_warnings=True)
    #
    #     observables = amici.assignmentRules2observables(
    #         sbml_importer.sbml,  # the libsbml model object
    #         filter_function=lambda variable: variable.getId().startswith(
    #             "observable_")
    #     )
    #     # print(observables)
    #
    #     # Sometimes get AttributeError: 'PosixPath' object has no attribute 'startswith'?
    #     sbml_importer.sbml2amici(model_name, model_dir,
    #                              constant_parameters=constant_parameters,
    #                              observables=observables,
    #                              compute_conservation_laws=False)
    #
    # # load the generated module
    # model_module = amici.import_model_module(model_name, model_dir)
    # # Create Model instance
    # model = model_module.getModel()
    #
    # print("Model parameters:", list(model.getParameterIds()))
    # print("Model outputs:   ", list(model.getObservableIds()))
    # print("Model states:    ", list(model.getStateIds()))
    #
    # # Now get data to do the fit
    # print()
    # # Get expr mat time
    # # TODO later on put this all in a class again
    # if expr_mat_time.aggregation_method == AggregationMethod.EIGENGENE:
    #     expressions: pd.DataFrame = expr_mat_time.df.groupby('cluster_id').apply(
    #         expr_mat_time._get_eigengene_over_time)
    #     # Add constant value to eigengenes
    #     expressions = abs(expressions.min().min()) + expressions
    # elif expr_mat_time.aggregation_method == AggregationMethod.MEAN:
    #     expressions: pd.DataFrame = expr_mat_time.df.groupby('cluster_id').apply(
    #         expr_mat_time._get_mean_over_time)
    #     # dataset.plot_clusters_over_time()
    # else:
    #     raise NotImplementedError
    #
    # for word in expr_mat_time.condition_names:
    #     # Deepcopy first?
    #     # dataset.keep_only_samples_with_string(word)
    #     valid_index = expressions.columns.get_level_values(
    #         'condition').isin(['zero', word])
    #     data = expressions.loc[:, valid_index]
    #     # Ensure that time is increasing
    #     data = data.sort_index(axis=1, level='time')
    #     # Convert time into hours
    #     time = data.columns.get_level_values('time') / pd.to_timedelta(1,
    #                                                                    unit='h')
    #     assert len(data.columns) == len(time)
    #     data = data.to_numpy()
    #     time = time.to_numpy()
    #
    #     # set timepoints for which we want to simulate the model
    #     model.setTimepoints(time)
    #
    #     # # Here: set u_t to correct value
    #     # TODO handle this correctly perhaps -> now kinda works for temp
    #     # TODO Check if different results for different temps
    #     custom_param_dict = {'21': 0,
    #                          '32': 1}
    #     model.setFixedParameterById('temp', custom_param_dict[word])
    #
    #     # set parameters to optimal values found in the benchmark collection
    #     # model.setParameterScale(amici.ParameterScaling.log10)
    #     nr_params = len(model.getParameterIds())
    #     # IF all zero, does not throw error
    #     # model.setParameters(np.zeros(nr_params))
    #
    #     # model.setParameters(np.random.standard_normal(nr_params)/10)
    #     some_params = np.random.rand(nr_params)/10
    #     model.setParameters(some_params)
    #
    #     # TODO is this needed? \/
    #     # model.setInitialStates()
    #
    #     # Create solver instance
    #     solver = model.getSolver()
    #     # Run simulation using model parameters from the benchmark collection and default solver options
    #     rdata = amici.runAmiciSimulation(model, solver)
    #
    #     plot_observable_trajectories(rdata)
    #     plt.show()
    #
    #     plt.plot(rdata.by_id('u_t'))
    #     plt.show()
    #
    #
    #     edata = amici.ExpData(
    #         data.shape[0],  # number of observables
    #         0,  # number of event outputs
    #         0,  # maximum number of events
    #         time,  # timepoints
    #     )
    #     # set observed data
    #     for i in range(data.shape[0]):
    #         edata.setObservedData(data[i,:], i)
    #
    #     rdata = amici.runAmiciSimulation(model, solver, edata)
    #
    #     print(f"chi2 value using AMICI: {rdata['chi2']}")
    #
    #     # we make some more adjustments to our model and the solver
    #     model.requireSensitivitiesForAllParameters()
    #
    #     solver.setSensitivityMethod(amici.SensitivityMethod.forward)
    #     solver.setSensitivityOrder(amici.SensitivityOrder.first)
    #
    #     objective = pypesto.AmiciObjective(
    #         amici_model=model, amici_solver=solver, edatas=[edata],
    #         max_sensi_order=1
    #     )
    #
    #     # the generic objective call
    #     print(f"Objective value: {objective(some_params)}")
    #     # a call returning the AMICI data as well
    #     obj_call_with_dict = objective(some_params, return_dict=True)
    #     print(
    #         f'Chi^2 value of the same parameters: {obj_call_with_dict["rdatas"][0]["chi2"]}'
    #     )
    #
    #     # So we can get objective function now, just have to proceed with that
    #     n_starts = 20  # usually a value >= 100 should be used
    #     engine = pypesto.engine.MultiProcessEngine()
    #     result = optimize.minimize(
    #         problem=problem,
    #         optimizer=optimizer,
    #         n_starts=n_starts,
    #         startpoint_method=startpoint_method,
    #         engine=engine,
    #         options=opt_options,
    #     )


