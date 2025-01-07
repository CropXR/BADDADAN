import copy
import logging
from pathlib import Path

import dill as pickle
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from GEOparse import get_GEO
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage
from fastcluster import linkage
from sklearn.metrics import adjusted_rand_score

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from analysis_pipelines import module_network_from_tf2_output
from data_wrangling import parse_go_enrichment_output
from exceptions import RegulatoryDisagreementError
from helpers import get_info_from_gse65046, keep_common_genes_in_dfs


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

def similarity_matrices_local_and_atted(expr_mat: ExpressionMatrixTimeSeries,
                                        atted_path: Path,
                                        out_path: Path):
    local_corr = expr_mat.get_similarity_matrix()
    # local_corr.to_csv(out_path / 'local_similarity.csv')
    atted_corr = pd.read_parquet(atted_path)
    atted_corr = atted_corr.set_index(atted_corr.columns[0])
    # atted_corr.to_csv(out_path / 'atted_all_genes_unnormalised.csv')

    atted_corr, local_corr, selected_genes = keep_common_genes_in_dfs(atted_corr,
                                                                      local_corr)

    expr_mat.df = expr_mat.df.loc[expr_mat.df.index.isin(selected_genes), :]
    expr_mat.df.to_csv(out_path / 'expression_de_atted_overlap_genes.csv')

    atted_score_flat = squareform(atted_corr, checks=False)
    sns.histplot(atted_score_flat)
    plt.show()

    atted_score_flat_norm = atted_score_flat - min(atted_score_flat)
    atted_score_flat_norm = atted_score_flat_norm / max(atted_score_flat_norm)

    sns.histplot(atted_score_flat_norm)
    plt.show()

    atted_score_norm = squareform(atted_score_flat_norm)
    atted_score_norm = pd.DataFrame(atted_score_norm, index=atted_corr.index, columns=atted_corr.columns)

    atted_score_norm.to_csv(out_path / 'atted_selected_genes_similarity.csv')
    # Plot both distributions
    local_dist_flat = squareform(local_dist)


    sns.histplot(local_dist_flat, binwidth=.2, element='step', fill=False, common_norm=False)
    sns.histplot(atted_dist_flat, binwidth=.2, element='step', fill=False, common_norm=False)
    plt.legend(['Local Distances', 'Atted Distances'])

    plt.savefig(out_path / 'raw_input_distances.png')
    plt.close()

    # Convert into z-scores
    local_dist_flat_norm = (local_dist_flat - np.mean(
        local_dist_flat)) / np.std(local_dist_flat)

    sns.histplot(local_dist_flat_norm, binwidth=.2, element='step', fill=False, common_norm=False)
    sns.histplot(atted_dist_flat_norm, binwidth=.2, element='step', fill=False, common_norm=False)
    plt.legend(['Local Distances', 'Atted Distances'])
    plt.savefig(out_path / 'normalised_distances.png')
    plt.close()


def combine_local_distance_and_prior(local_dist: pd.DataFrame,
                                     prior_score: pd.DataFrame,
                                     dists_out_path: Path,
                                     combo: str ='sum',
                                     calculate_linkages: bool = True,
                                     plot_out_path: Path | None = None):
    """Get combo of local distances and atted_distances to
    do distances simulatenously

    :param local_dist: local distances as dataframe
    :param prior_score: prior scores (e.g. from atted)
    :param dists_out_path: file path to save combined distances (should be a name that ends in .parquet.gzip)
    :param combo: how to combine the distances (sum vs minimum of either)
    :param calculate_linkages: If true, calculate the linkages for later use
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
    if calculate_linkages:
        dist_dict = {'atted': atted_dist_flat_norm,
                     'combined': combined_distances,
                     'local': local_dist_flat_norm}

        for input_dist_name, input_dists in dist_dict.items():
            logging.info(f'For input {input_dist_name}')
            if np.min(input_dists) < 0:
                input_dists = input_dists - np.min(input_dists)
            for method in ['complete', 'single', 'average']:
                logging.info(f'Performing {method} now')
                linkage_matrix = linkage(input_dists, method=method)
                np_path = out_path / f'{input_dist_name}_distances_{method}_linkage.npy'
                np.save(np_path, linkage_matrix)
    return combined_dist_df
    # sns.clustermap(squareform(atted_dist_flat_norm),
    #                row_linkage=np.load(out_path / f'atted_distances_complete_linkage.npy'),
    #                col_linkage=np.load(out_path / f'atted_distances_complete_linkage.npy'),
    #                )
    # plt.show()

    # sns.clustermap(squareform(combined_distances),
    #                row_linkage=np.load(out_path / f'combined_distances_average_linkage.npy'),
    #                col_linkage=np.load(out_path / f'combined_distances_average_linkage.npy'),
    #                )
    # plt.show()



def rand_index_both_clusterings(tf2_input_1, tf2_input_2):
    df1 = pd.read_csv(tf2_input_1, sep=' ', names=['cluster_id_1', 'gene_name'])
    df2 = pd.read_csv(tf2_input_2, sep=' ', names=['cluster_id_2', 'gene_name'])

    merged_df = df1.merge(df2, on='gene_name', how='inner')
    clustering_1 =merged_df['cluster_id_1'].tolist()
    clustering_2 =merged_df['cluster_id_2'].tolist()
    adjusted_rand_score(clustering_1, clustering_2)


def how_many_cignet_modules_tfbs_enriched(modules_path: Path, tf2_network_output: Path):
    """See how many of the input modules have a significant TFBS enrichment"""
    modules_df = pd.read_csv(modules_path, sep='\t')
    tfbs_df = pd.read_csv(tf2_network_output, sep='\t')
    print(f"Modules in original {modules_df['moduleID'].nunique()}")
    print(f"Modules with >0 tfbs: {tfbs_df['GeneSet'].nunique()}")


def see_how_intermodular_network_looks(modules_path: Path, tf2_network_output: Path, cytoscape_output_path: Path):
    """From tf2network output and module assignments, create intermodular network."""
    my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(tf2_network_output, q_value_cutoff=0.01)
    my_grn.add_tf_module_mappings(modules_path, from_tf2_input=True)
    my_grn.clean_up_network()
    intermodular_network = my_grn.get_module_module_network(tf_can_be_from_multiple_modules=True)
    save_path = str(cytoscape_output_path)
    intermodular_network.save_for_cytoscape(save_path)
    intermodular_network.plot_network(with_labels=False)

def create_intermodular_network_first_try(modules_path: Path, tf2_network_output: Path, out_path: Path):
    """Some first try at creating a intermodular network. Not majorly important"""
    modules_df = pd.read_csv(modules_path, sep='\t')
    tfbs_df = pd.read_csv(tf2_network_output, sep='\t')
    print()
    merged_df = modules_df.merge(tfbs_df, how='left', left_on='moduleID', right_on='GeneSet')
    # Keep only relevant columns
    merged_df = merged_df[['moduleID', 'geneID', 'Regulator']]
    # See if regulator is also present in other module
    merged_df['has_link'] = merged_df['Regulator'].isin(merged_df['geneID'])
    merged_df = merged_df[merged_df['has_link'] == True]
    # Add regulator's module
    regulator_id = merged_df.merge(merged_df, how='left', left_on='Regulator', right_on='geneID')
    module_links = regulator_id[['moduleID_x', 'moduleID_y']]
    no_dupes = module_links.drop_duplicates(ignore_index=True)
    no_dupes = no_dupes.dropna()
    print(len(no_dupes))
    # Save for cytoscape?
    merged_df.to_csv(out_path, index=False)

def compare_annotations(soft_path, csv_path):
    """See if gene annotations differ between doing the SOFT-based annotation
    or annotation_file csv based annotation

    :param soft_path: path to soft geo input
    :param csv_path: path to annotation
    """

    # Data preparation
    csv_df = pd.read_csv(csv_path, sep=';', na_values='No match')
    csv_df = csv_df[['CATMA_ID', 'AGI_code_Spring_2004']]
    csv_df.columns = ['ID', "CSV_MAP"]
    csv_df['ID'] = csv_df['ID'].str.upper()
    csv_df['CSV_MAP'] = csv_df['CSV_MAP'].str.upper()

    geo_file = get_GEO(filepath=str(soft_path))
    geo_names = geo_file.gsms['GSM1586359'].table.ID_REF.str.upper()

    soft_df = geo_file.gpls['GPL16132'].table
    soft_df = soft_df[['ID', 'ORF']]
    soft_df.columns = ['ID', 'SOFT_MAP']
    soft_df['ID'] = soft_df['ID'].str.upper()
    soft_df['SOFT_MAP'] = soft_df['SOFT_MAP'].str.upper()

    master_df = csv_df.merge(soft_df, on='ID', how='outer')
    # Focus on genes that are in the input data
    master_df = master_df[master_df.ID.isin(geo_names)]
    print('Genes at start')
    print(len(master_df))

    print('Genes that map to nan on both')
    is_na_df = master_df[['CSV_MAP', 'SOFT_MAP']].isna()
    both_nan_df = is_na_df.all(axis=1)
    print(sum(both_nan_df))

    at_least_one_map_df = master_df[~both_nan_df]
    print('Genes that map at least once')
    print(len(at_least_one_map_df))

    print('Genes that map exactly once')
    print(len(master_df[master_df['CSV_MAP'].isna() ^ master_df['SOFT_MAP'].isna()]))

    print('Only soft maps')
    print(len(master_df[master_df['CSV_MAP'].isna() & ~master_df['SOFT_MAP'].isna()]))

    print('Only csv maps')
    print(len(master_df[master_df['SOFT_MAP'].isna() & ~master_df['CSV_MAP'].isna()]))

    print('Both map')
    both_map_df = master_df[~master_df['SOFT_MAP'].isna() & ~master_df['CSV_MAP'].isna()]
    print(len(both_map_df))

    print('Mappings agree')
    print(sum(both_map_df['SOFT_MAP'] == both_map_df['CSV_MAP']))

    print('Mappings disagree')
    print(sum(both_map_df['SOFT_MAP'] != both_map_df['CSV_MAP']))


def see_expression_genes_of_interest(exp_mat_path: Path):
    exp_mat = ExpressionMatrixTimeSeries.from_geo_file(exp_mat_path,
                                                       annotate_from_gpl=True,
                                                       log2_transform=True)
    genes_of_interest = ["AT1G30100",
                        "AT1G31800",
                        "AT1G52340",
                        "AT1G78390",
                        "AT2G27150",
                        "AT3G14440",
                        "AT3G24220",
                        "AT4G18350",
                        "AT4G19170",
                        "AT4G25700",
                        "AT5G52570",
                        "AT5G67030"]

    #BES1 & HY5
    genes_of_interest = ['AT1G19350', 'AT5G11260']

    # Some genes from the paper
    genes_of_interest = ["AT4G22880", "AT1G56650", "AT5G13930" ]


    exp_mat.df = exp_mat.df[exp_mat.df.index.isin(genes_of_interest)]

    exp_mat.df['cluster_id']= [i for i, _ in enumerate(genes_of_interest, 1)]
    exp_mat.has_been_clustered = True
    exp_mat.column_parser = get_info_from_gse65046

    expr_mat_drought = copy.deepcopy(exp_mat)
    expr_mat_drought.keep_only_samples_with_string('drought')
    expr_mat_drought.plot_clusters_over_time()
    # expr_mat_drought.plot_clusters_over_time(title='Drought', plot_units=True)

    expr_mat_control = copy.deepcopy(exp_mat)
    expr_mat_control.keep_only_samples_with_string('control')
    expr_mat_control.plot_clusters_over_time()
    # expr_mat_control.plot_clusters_over_time(title='Control', plot_units=True)
    print()

def analyse_all_go_enrichments(in_dir: Path):
    """See how many modules have a signigicant BP go enrichment"""
    df_list = []
    total_modules = len(list(in_dir.iterdir()))
    print(total_modules)
    for file in in_dir.iterdir():
        df_list.append(parse_go_enrichment_output(file))
    master_df = pd.concat(df_list)
    group_df = master_df.groupby('module_name')
    print(group_df.ngroups)
    print(group_df.ngroups / total_modules)


def get_coex_from_tf2_output(input_file):
    """From TF2Network output, see the coexpression scores"""
    df = pd.read_csv(input_file, sep='\t')
    df_grp = df.groupby('GeneSet')
    logging.info(f'input: {input_file}\n {df_grp.ngroups} groups\n')
    mean_coex = df_grp['CO'].mean()
    return mean_coex


def compare_modules_to_local_modules_with_tfbs(expr_mat_dict, local_dists_tf2_output):
    """For the modules that are found in local distance and have a TFBS enriched
    See if they overlap with modules you find from other clusterings.
    Could suggest that you only find these modules using local clustering
     because they are highly specific to this stress and other clusterings
     find modules that are present in other contexts (and thus cannot be
     found through local clusterings,
    """
    local_df = expr_mat_dict['local_dists'].df
    atted_df = expr_mat_dict['atted_only'].df
    summed_dist_df = expr_mat_dict['summed_dists'].df
    summed_dist_df = summed_dist_df.rename(
        columns={'cluster_id': 'cluster_id_summed'})
    tf2_out_local = pd.read_csv(
        local_dists_tf2_output,
        sep='\t')
    has_tf_enriched_index = local_df['cluster_id'].isin(
        tf2_out_local['GeneSet'])
    local_df = local_df[has_tf_enriched_index]
    merged_df = atted_df.join(local_df, how='inner', lsuffix='_atted',
                              rsuffix='_local')
    merged_df = merged_df.join(summed_dist_df, how='inner')
    merged_df = merged_df.loc[:, merged_df.columns.str.contains('cluster_id')]
    print(adjusted_rand_score(merged_df['cluster_id_atted'],
                              merged_df['cluster_id_local']))
    print(adjusted_rand_score(merged_df['cluster_id_summed'],
                              merged_df['cluster_id_local']))


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


def get_robustness_random_modules(jackknife_paths, full_dataset_path, figure_out_dir):
    out_list = []
    module_size_list = []

    # Full dataset
    full_dataset = pd.read_csv(full_dataset_path, usecols=[1, 2])
    full_dataset = full_dataset.set_index('gene_id')
    # Module sizes (in full dataset)
    module_size_list.extend(
        [size for size in full_dataset.value_counts().to_list()]
    )

    # Robustness
    for jackknife_path in jackknife_paths:
        subset = pd.read_csv(jackknife_path, usecols=[1, 2])
        subset = subset.set_index('gene_id')

        merged_df = subset.join(full_dataset,
                                how='inner',
                                lsuffix='_subset',
                                rsuffix='_og')
        ari = adjusted_rand_score(merged_df['colors_subset'],
                                  merged_df['colors_og'])
        out_list.append(('robustness_merged_dists', ari))

        # Do random clustering now
        subset['colors'] = np.random.permutation(subset['colors'] )
        # Merge genes
        merged_df = subset.join(full_dataset,
                                how='inner',
                                lsuffix='_subset',
                                rsuffix='_og')
        ari = adjusted_rand_score(merged_df['colors_subset'],
                                  merged_df['colors_og'])
        out_list.append(('robustness_random', ari))

    df = pd.DataFrame.from_records(out_list,
                                   columns=['Method', 'Robustness'])
    sns.boxplot(data=df, y='Robustness', x='Method', hue='Method')
    plt.ylim((0, .9))
    plt.savefig(figure_out_dir / 'boxplot_robustness_with_random_modules.png')
    plt.close()

    sns.swarmplot(data=df, y='Robustness', x='Method', hue='Method')
    plt.ylim((0, .9))
    plt.savefig(figure_out_dir / 'swarmplot_robustness_with_random_modules.png')
    plt.close()
    return out_list


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
        plt.ylabel("Salicylic Acid Level (SA)")
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
