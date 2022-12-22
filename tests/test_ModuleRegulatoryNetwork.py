from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from ModuleRegulatoryNetwork import ModuleRegulatoryNetwork


def test_import_network_edges_from_lpan():
    path_to_network_edges = Path('../data/aracne_network_edges.csv')
    path_to_orignal_cluster = Path('../data/my_clustering_edgelist.csv')
    my_graph = ModuleRegulatoryNetwork.from_lpan_edge_csv(path_to_network_edges,
                                                          top_rank=30)
    my_graph.add_tf_module_mappings(path_to_orignal_cluster)
    my_graph.clean_up_network()
    my_graph.plot_network()
