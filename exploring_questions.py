import copy
from pathlib import Path

import pandas as pd
from GEOparse import get_GEO

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from data_wrangling import parse_go_enrichment_output
from helpers import get_info_from_gse65046


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

