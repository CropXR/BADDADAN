import logging
from pathlib import Path

import numpy as np
import pandas as pd


def merge_ath_annotation_for_goatools(in_path: Path, out_path: Path):
    """Take go annotation from TAIR, and transform it so it can be read by GOATOOLS
    """
    column_names = [
        "locus name",
        "TAIR accession",
        "object name",
        "relationship type",
        "GO term",
        "GO ID",
        "TAIR Keyword ID",
        "Aspect",
        "GOslim term",
        "Evidence code",
        "Evidence description",
        "Evidence with",
        "Reference",
        "Annotator",
        "Date annotated",]
    df = pd.read_csv(in_path, sep='\t', comment='!', names=column_names)
    df_group = df.groupby('locus name')
    out_df = df_group['GO ID'].apply(lambda x: ';'.join(x))
    out_df.to_csv(out_path, sep='\t')


def parse_go_enrichment_output(in_file: Path, cutoff: float = 0.05) -> pd.DataFrame:
    """Take go enrichment output, and select only BP annotations with a fdr-corrected p-value of <0.05"""
    logging.info(f'Parsing {in_file}')
    df = pd.read_csv(in_file, sep='\t')
    # FDR correction
    df = df[df['p_fdr_bh'] < cutoff]
    # ONLY BP
    df = df[df['NS'] == 'BP']
    df['module_name'] = in_file.stem
    return df

def create_correlation_matrix_from_atted_ii_raw(corr_file: Path, gene_ids: Path):
    """From pairwise correlations of atted_ii, create one expressionmatrix as pandas dataframe

    :param corr_file: path to file that contains atted_ii correlations in the form of
        10723023:818113 7.9030
        10723023:835497 6.6774
        10723023:816702 6.3223
        first item gene 1, second gene 2, third item the correlation value
    :param gene_ids: list of all gene ids, will be used to construct the
                     dataframe
    :return: Dataframe with all values
    """
    with gene_ids.open('r') as f:
        gene_names = f.read().split()
    corr_matrix = pd.DataFrame(index=gene_names, columns=gene_names)
    with corr_file.open('r') as f:
        all_lines = f.readlines()
        for i, line in enumerate(all_lines):
            if i % 1_000_000 == 0:
                print(f'{i/len(all_lines):.2%}')
            line = line.strip()
            connection, corr_value = line.split()
            gene_1, gene_2 = connection.split(':')
            corr = corr_matrix.at[gene_1, gene_2]
            if np.isnan(corr):
                corr_matrix.at[gene_1, gene_2] = corr
                corr_matrix.at[gene_2, gene_1] = corr

    return corr_matrix

