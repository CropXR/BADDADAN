from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


class ModuleRegulatoryNetwork:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        # Standard prefixes
        self.tf_prefix = 'TF'
        self.module_prefix = 'MODULE'

    @classmethod
    def from_lpan_edge_csv(cls, lpan_file_path: Path, top_rank: int = None) -> ModuleRegulatoryNetwork:
        """Create object from lpan edge csv.

        :param lpan_file_path: Path to csv output by LPAN.
        E.g. aracne_network_edges.csv
        :param top_rank: If provided, keep only the top_rank  number of
        connections. I.e. the strongest one predicted by lpan.
        """
        some_df = pd.read_csv(lpan_file_path)
        if top_rank:
            some_df = some_df[some_df['rank'] < top_rank]
        some_df['origin'] = 'binds_to'
        a_graph = nx.from_pandas_edgelist(some_df, source='regulator',
                                          target='target',
                                          edge_attr='origin',
                                          create_using=nx.DiGraph)
        return cls(a_graph)

    def plot_network(self, draw_func: Callable = nx.draw_kamada_kawai, out_path: Path = None):
        """Plot network using matplotlib. Colour mean something, but I'll document that later"""
        node_color_map = ['blue' if ('TF' in node) else 'orange' for node in
                     self.graph.nodes]
        edge_color_map = ['black' if ('transcribed_by' in atrs.get('origin'))
                          else 'brown' for _, _, atrs in self.graph.edges(data=True)]

        draw_func(self.graph, node_color=node_color_map,
                  edge_color=edge_color_map)
        if out_path:
            plt.savefig(out_path)
        else:
            plt.show()

    def add_tf_module_mappings(self, path_to_orignal_cluster: Path) -> None:
        """Include what TFs are transcribed by which cluster.
        Add these edges with this method, input file is created from ExpressionMatrix (I think)."""
        original_connections = nx.read_edgelist(path_to_orignal_cluster, create_using=nx.DiGraph)
        # Invert connections, because <TF:is transcribed by:Module>
        original_connections = original_connections.reverse()
        self.graph.add_edges_from(original_connections.edges, origin='transcribed_by')

    def clean_up_network(self) -> None:
        """Remove all non-binding TFs and unused modules."""
        self.remove_non_binding_tfs()
        self.remove_unused_modules()

    def remove_non_binding_tfs(self) -> None:
        """Delete transcription factors from the graph that do not bind to
        any module. I.e. that have an out_degree of 0.
        """
        # Find nodes which are tf and have out degree 0
        non_binding_tfs = []
        for node_name, out_degree in self.graph.out_degree():
            if self.tf_prefix in node_name and out_degree == 0:
                non_binding_tfs.append(node_name)
        # Remove them
        self.graph.remove_nodes_from(non_binding_tfs)

    def remove_unused_modules(self) -> None:
        """Delete modules which are unused. I.e. modules that are not regulated
         by any transcription factor, and also don't encode for any transcription factors"""
        unused_modules = []
        for node_name in self.graph.nodes:
            if (self.module_prefix in node_name
                    and self.graph.in_degree(node_name)
                    == self.graph.out_degree(node_name)
                    == 0):
                unused_modules.append(node_name)
        # Remove them
        self.graph.remove_nodes_from(unused_modules)
