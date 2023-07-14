import logging
from pathlib import Path

from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from DynamicModels.OdeModel import OdeModel

from main import compare_clusterings, camila_red_panda


def test_get_tf2_input(my_tf2_input):
    pass

def test_camila_red_panda():
    in_path = Path('../data/time_series_datasets/GSE5628_family.soft')
    out_dir = Path('../data/time_series_datasets/tf2network_approach/test_output/')

    camila_red_panda(in_path, out_dir, True, 'mad', 2000, 4)

def test_compare_clusterings():
    # log2_cluster = Path('../data/time_series_datasets/tf2network_approach/2000_highest_qcd_log2/01_tf2network_input_2000_highest_qcd_genes.txt')
    # raw_cluster = Path('../data/time_series_datasets/tf2network_approach/2000_highest_qcd_no_log2/01_tf2network_input_2000_highest_qcd_genes.txt')
    for method in ['mad', 'cv', 'qcd']:
        logging.info(f'Method= {method}')
        log2_cluster = Path(
            f'../data/time_series_datasets/tf2network_approach/2000_highest_{method}_log2/01_tf2network_input_2000_highest_{method}_genes.txt')
        raw_cluster = Path(
            f'../data/time_series_datasets/tf2network_approach/2000_highest_{method}_no_log2/01_tf2network_input_2000_highest_{method}_genes.txt')
        compare_clusterings(log2_cluster, raw_cluster)

