import logging
from pathlib import Path

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
