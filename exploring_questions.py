import copy
from pathlib import Path

import pandas as pd
from GEOparse import get_GEO

from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from helpers import get_info_from_gse65046


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
