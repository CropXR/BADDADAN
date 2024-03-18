import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from GEOparse import get_GEO
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import adjusted_rand_score

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from data_wrangling import parse_go_enrichment_output
from helpers import get_info_from_gse65046

def save_local_distance_matrix(soft_file_path: Path, do_log_2: bool,
                               atted_path: Path, out_path: Path):
    """Calculate local pairwise distance matrix
    
    :param soft_file_path: Path to input gene expression file
    :param do_log_2: If true, perform log2-transformation to gene expressions
    :param out_path: Path to save distance matrix (should be .pkl file)
    """
    expr_mat_time: ExpressionMatrixTimeSeries = ExpressionMatrixTimeSeries.from_geo_file(
        soft_file_path,
        log2_transform=do_log_2,
        annotate_from_gpl=True
    )
    expr_mat_time.column_parser = get_info_from_gse65046
    expr_mat_time.merge_biological_samples()
    expr_mat_time.save_distance_matrix(out_path)
    dist_local = expr_mat_time.get_distance_matrix()
    print()

def sum_local_distance_and_atted(local_dist_path: Path, atted_path: Path,
                                 out_path: Path):
    """Get sum of local distances and atted_distances to
    do distances simulatenously"""
    atted_score = pd.read_parquet(atted_path)
    atted_score = atted_score.set_index(atted_score.columns[0])
    local_dist = pd.read_pickle(local_dist_path)

    # toy_size = 5000
    # atted_score = atted_score.iloc[:toy_size, :toy_size]
    # local_dist = local_dist.iloc[:toy_size, :toy_size]

    # get intersection
    selected_genes = atted_score.index.intersection(local_dist.index)
    # Shrink dataframes so match in size
    local_dist = local_dist.loc[selected_genes, selected_genes]
    atted_score = atted_score.loc[selected_genes, selected_genes]
    # Higher score -> lower dist
    atted_dist = atted_score.max().max() - atted_score
    assert (local_dist.index.equals(
        atted_dist.index) and local_dist.columns.equals(atted_dist.columns))

    # Plot both distributions
    local_dist_flat = squareform(local_dist)
    atted_dist_flat = squareform(atted_dist, checks=False)

    sns.histplot(local_dist_flat, binwidth=.2, element='step', fill=False)
    sns.histplot(atted_dist_flat, binwidth=.2, element='step', fill=False)
    plt.savefig(out_path / 'raw_input_distances.png')
    plt.close()

    # Convert into z-scores
    local_dist_flat_norm = (local_dist_flat - np.mean(
        local_dist_flat)) / np.std(local_dist_flat)
    atted_dist_flat_norm = (atted_dist_flat - np.mean(
        atted_dist_flat)) / np.std(atted_dist_flat)
    sns.histplot(local_dist_flat_norm, binwidth=.2, element='step', fill=False)
    sns.histplot(atted_dist_flat_norm, binwidth=.2, element='step', fill=False)
    plt.savefig(out_path / 'normalised_distances.png')
    plt.close()

    summed_distances = local_dist_flat_norm + atted_dist_flat_norm

    sns.histplot(summed_distances, binwidth=.2, element='step', fill=False)
    plt.savefig(out_path / 'summed_distances.png')
    plt.close()

    square_summed_distances = squareform(summed_distances)
    summed_dist_df = pd.DataFrame(data=square_summed_distances,
                                  index=local_dist.index,
                                  columns=local_dist.index)

    summed_dist_df.to_parquet(out_path / 'atted_local_dist_summed.parquet.gzip',
                              compression='gzip')

    for method in ['complete', 'single', 'average']:
        print(f'Performing {method} now')
        linkage_matrix = linkage(summed_distances, method=method)
        np_path = out_path / f'summed_distances_{method}_linkage.npy'
        np.save(np_path, linkage_matrix)


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
                                                        log2_transform=True,
                                                             annotate_from_gpl=True)
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
    expr_mat_drought.plot_clusters_over_time(title='Drought')
    # expr_mat_drought.plot_clusters_over_time(title='Drought', plot_units=True)

    expr_mat_control = copy.deepcopy(exp_mat)
    expr_mat_control.keep_only_samples_with_string('control')
    expr_mat_control.plot_clusters_over_time(title='Control')
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

def exploratory_module_selection(input_file, modules_file):
    """Get an idea of the characteristics of the gene modules"""
    expr_mat_time: ExpressionMatrixTimeSeries = ExpressionMatrixTimeSeries.from_geo_file(
        input_file,
        log2_transform=True,
        annotate_from_gpl=True)
    cignet_modules = Path('data/resources/cig_data/ModuleGenes_two_cols.txt')
    expr_mat_time.assign_clusters_from_cignet_file(cignet_modules, remove_dupes=True)
    expr_mat_time.show_characteristics_of_clusters()

def get_coex_from_tf2_output(input_file):
    """From TF2Network output, see the coexpression scores"""
    df = pd.read_csv(input_file, sep='\t')
    df_grp = df.groupby('GeneSet')
    logging.info(f'input: {input_file}\n {df_grp.ngroups} groups\n')
    mean_coex = df_grp['CO'].mean()
    return mean_coex