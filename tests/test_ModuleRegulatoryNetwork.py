from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from ModuleRegulatoryNetwork import ModuleRegulatoryNetwork


def test_import_network_edges_from_lpan():
    some_p_val_cutoff = 0.05
    path_to_network_edges = Path('../data/aracne_network_edges.csv')
    path_to_orignal_cluster = Path('../data/my_clustering_edgelist.csv')
    my_graph = ModuleRegulatoryNetwork.from_lpan_edge_csv(path_to_network_edges,
                                                          top_rank=10)
    my_graph.add_tf_module_mappings(path_to_orignal_cluster)
    my_graph.plot_network()
    # TODO you need to debug this because the colours don't always make sense.
    plt.show()
