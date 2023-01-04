"""
Creates expression annotation and expression matrix that can be used by
other tests
"""

from pathlib import Path

import pandas as pd
import logging
import GEOparse
import pytest

from ExpressionArrayAnnotation import ExpressionArrayAnnotation
from ExpressionMatrix import ExpressionMatrix, ExpressionMatrixTimeSeries
from ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from OdeInference import OdeInference

pd.options.display.width = 0
GEOparse.logger.set_verbosity('INFO')
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def my_expression_annotation():
    my_path = Path('../data/affy_ATH1_array_elements-2010-12-20.txt')
    return ExpressionArrayAnnotation(my_path)


@pytest.fixture
def my_expression_matrix(my_expression_annotation: ExpressionArrayAnnotation):
    my_expression = Path('../data/GSE15689_family.soft')
    expr_mat = ExpressionMatrix.from_geo_file(my_expression,
                                              my_expression_annotation)
    return expr_mat


@pytest.fixture
def my_time_series_expressions(my_expression_annotation: ExpressionArrayAnnotation) -> ExpressionMatrixTimeSeries:
    time_series_expressions = Path(
        '../data/GSE5628_family.soft')
    expr_mat_time = ExpressionMatrixTimeSeries.from_geo_file(
        time_series_expressions, my_expression_annotation, log2_transform=True)
    return expr_mat_time

@pytest.fixture
def my_grn() -> ModuleRegulatoryNetwork:
    path_to_network_edges = Path('../data/aracne_network_edges.csv')
    path_to_orignal_cluster = Path('../data/my_clustering_edgelist.csv')
    my_graph = ModuleRegulatoryNetwork.from_lpan_edge_csv(path_to_network_edges,
                                                          top_rank=25)
    my_graph.add_tf_module_mappings(path_to_orignal_cluster)
    return my_graph

@pytest.fixture
def my_module_module_network(my_grn: ModuleRegulatoryNetwork) -> ModuleRegulatoryNetwork:
    my_grn.clean_up_network()
    # my_grn.plot_network()
    my_module_network = my_grn.get_module_module_network()
    # my_module_network.plot_network()
    return my_module_network

@pytest.fixture
def my_ode(my_module_module_network: ModuleRegulatoryNetwork) -> OdeInference:
    return my_module_module_network.convert_to_ode()

