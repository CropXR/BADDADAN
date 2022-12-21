from __future__ import annotations

import json
from pathlib import Path
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


class ModuleRegulatoryNetwork:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @classmethod
    def from_lpan_edge_csv(cls, lpan_file_path: Path, p_value_cutoff: float = 0.05, top_rank: int = None):
        """Create object from lpan edge csv"""
        some_df = pd.read_csv(lpan_file_path)
        if top_rank:
            some_df = some_df[some_df['rank'] < top_rank]
        some_df['origin'] = 'binds_to'
        a_graph = nx.from_pandas_edgelist(some_df, source='regulator',
                                          target='target',
                                          edge_attr='origin',
                                          create_using=nx.DiGraph)
        return cls(a_graph)

    def plot_network(self):
        node_color_map = ['blue' if ('TF' in node) else 'orange' for node in
                     self.graph.nodes]
        edge_color_map = ['black' if ('transcribed_by' in atrs.get('origin'))
                          else 'brown' for _, _, atrs in self.graph.edges(data=True)]

        nx.draw(self.graph, node_color=node_color_map,
                edge_color=edge_color_map)

    def add_tf_module_mappings(self, path_to_orignal_cluster: Path) -> None:
        original_connections = nx.read_edgelist(path_to_orignal_cluster)
        self.graph.add_edges_from(original_connections.edges, origin='transcribed_by')
