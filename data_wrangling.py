import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries


def merge_ath_annotation_for_goatools(in_path: Path,
                                      evidence_code_filter: bool | list[
                                          str] = False):
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
    if evidence_code_filter:
        assert len(evidence_code_filter) > 0, \
            f"No evidence codes provided: {evidence_code_filter=}"
        df = df[df['Evidence code'].isin(evidence_code_filter)]

    df_group = df.groupby('locus name')
    out_df = df_group['GO ID'].apply(lambda x: ';'.join(x))
    return out_df

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

def entrez_to_tair_id(annotation_path: Path, coexpression_matrix_path: Path):
    """Convert entrez gene ID columns and rows in dataframe to locus tag IDs

    :param annotation_path: Path to dataframe that maps GeneID to LocusTag
    (and contains these column names)
    :param coexpression_matrix_path: Matrix for which row and column names
    should be relabeled
    :return: Saves renamed csv as {coexpresssion_matrix_path}_renamed.csv
    """
    out_path = coexpression_matrix_path.with_stem(coexpression_matrix_path.stem + '_renamed')
    logging.info(f'{annotation_path} + {coexpression_matrix_path} -> {out_path}')
    df_annot = pd.read_csv(annotation_path, sep='\t')
    translation_dict = df_annot.set_index('GeneID')['LocusTag'].to_dict()
    df_coexp = pd.read_csv(coexpression_matrix_path, sep=',', index_col=0)
    df_coexp.columns = df_coexp.columns.astype(int)
    df_coexp_rename = df_coexp.rename(index=translation_dict, columns=translation_dict)
    df_coexp_rename.to_csv(out_path)


def module_network_from_tf2_output(expr_mat_time : ExpressionMatrixTimeSeries,
                                   tf2_in_path: Path,
                                   tf2_out_path: Path,
                                   threshold: float,
                                   module_plot_path: Path
                                   ) -> ModuleRegulatoryNetwork:
    """From TF2Network output, generate a module module network

    :param expr_mat_time: Expression matrix time series
    :param tf2_in_path: Path to TF2Network input
    :param tf2_out_path: Path to TF2Network output
    :param threshold: Up/down regulation threshold
    :param module_plot_path: Path where to plot output module network
    :return: Module module network
    """
    my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(tf2_out_path)
    my_grn.add_tf_module_mappings(tf2_in_path,
                                  from_tf2_input=True)
    my_grn.keep_only_modules_of_interest(expr_mat_time)
    my_grn.clean_up_network()
    my_grn.check_if_tfs_created_by_module(expr_mat_time,
                                          do_plotting=False,
                                          remove_low_corr=True,
                                          assert_correlated=False,
                                          corr_cutoff=.3)
    my_grn.set_up_or_downregulation(expr_mat_time, do_plotting=False,
                                    threshold=threshold)
    # my_grn.plot_network(nx.d  raw_kamada_kawai, with_labels=False)
    module_module = my_grn.get_module_module_network()
    # # module_module.graph = nx.create_empty_copy(module_module.graph, with_data=False)
    if module_plot_path is not None:
        module_module.plot_network(nx.draw_kamada_kawai , with_labels=True, out_path=module_plot_path)
    logging.info(list(module_module.graph.edges(data=True)))
    return module_module
