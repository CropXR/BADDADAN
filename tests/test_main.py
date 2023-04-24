import logging
from pathlib import Path

from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from DynamicModels.OdeModel import OdeModel

from main import fit_ode_to_data, thickening_thinning, \
    annotate_microarray_expression, compare_clusterings, count_flowering_genes


def test_get_tf2_input(my_tf2_input):
    pass


def test_tf2_inference(
        my_tf2_network,
        my_tf2_input: ExpressionMatrixTimeSeries):
    fit_ode_to_data(my_tf2_network, my_tf2_input)


def test_tf2_inference_with_extra_connection(
        my_tf2_network,
        my_time_series_expressions: ExpressionMatrixTimeSeries):
    # TODO how do we know for sure if the modules did not get mixed up?
    """Do inference on original dataset with model that contains additional connections"""
    my_tf2_network.add_regulator_to_module(target_module_idx=1,
                                           origin_module_idx=3)
    my_tf2_network.add_regulator_to_module(target_module_idx=3,
                                           origin_module_idx=1)
    fit_ode_to_data(my_tf2_network, my_time_series_expressions)


def test_tf2_inference_with_fewer_connection(
        my_tf2_network,
        my_time_series_expressions: ExpressionMatrixTimeSeries):
    """Do inference on original dataset with model that contains one fewer connection"""
    my_tf2_network.remove_regulator_from_module(target_module_idx=2,
                                                origin_module_idx=1)
    fit_ode_to_data(my_tf2_network, my_time_series_expressions)


def test_thickening_thinning(
        my_tf2_network,
        my_time_series_expressions: ExpressionMatrixTimeSeries
):
    thickening_thinning(my_tf2_network, my_time_series_expressions,
                        std_cutoff=1)


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


def test_count_flowering_genes():
    # Path(f'../data/time_series_datasets/tf2network_approach/2000_highest_{method}_log2/01_tf2network_input_2000_highest_{method}_genes.txt')
    cluster_dir = Path(f'../data/time_series_datasets/tf2network_approach/')
    for selected_gene_path in cluster_dir.glob('*/01*.txt'):
        df = count_flowering_genes(
            selected_gene_path,
            Path('../data/resources/flor_id_flowering_genes.pkl')
        )
        logging.info(f'{selected_gene_path.parts[-2]}')
        logging.info(f'Flowering genes: {len(df)}')
