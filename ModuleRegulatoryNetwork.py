from __future__ import annotations

from pathlib import Path
from typing import Callable

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from OdeInference import OdeInference


class ModuleRegulatoryNetwork:
    # Standard prefixes
    tf_prefix = 'TF'
    module_prefix = 'MODULE'

    # Descriptors used in graph to describe function of edge
    id_of_binding = 'binds_to'
    id_of_transcription = 'transcribed_by'
    # This is used for module-module interactions
    id_of_regulation = 'regulates'

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def plot_network(self, node_color_map: list = None,
                     draw_func: Callable = nx.draw_kamada_kawai,
                     out_path: Path = None):
        """Plot network using matplotlib."""

        node_color_map = ['blue' if (node in self.get_tfs()) else 'orange'
                          for node in self.graph.nodes]

        # TODO convert to an ENUM at some point?
        edge_mapping_dict = {self.id_of_regulation: 'green',
                             self.id_of_binding: 'brown',
                             self.id_of_transcription: 'black'}

        edge_color_map = [edge_mapping_dict[origin] for _, _, origin
                          in self.graph.edges.data('origin')]

        draw_func(self.graph, node_color=node_color_map,
                  edge_color=edge_color_map)
        if out_path:
            plt.savefig(out_path)
        else:
            plt.show()

    @classmethod
    def from_lpan_edge_csv(cls, lpan_file_path: Path, top_rank: int = None):
        """Create object from lpan edge csv.

        :param lpan_file_path: Path to csv output by LPAN.
        E.g. aracne_network_edges.csv
        :param top_rank: If provided, keep only the top_rank  number of
        connections. I.e. the strongest one predicted by lpan.
        """
        some_df = pd.read_csv(lpan_file_path)
        if top_rank:
            some_df = some_df[some_df['rank'] < top_rank]
        some_df['origin'] = cls.id_of_binding
        a_graph = nx.from_pandas_edgelist(some_df, source='regulator',
                                          target='target',
                                          edge_attr='origin',
                                          create_using=nx.DiGraph)
        return cls(a_graph)

    def add_tf_module_mappings(self, path_to_orignal_cluster: Path) -> None:
        """Include what TFs are transcribed by which cluster.
        Add these edges with this method, input file is created from
        ExpressionMatrix object.
        """
        original_connections = nx.read_edgelist(path_to_orignal_cluster,
                                                create_using=nx.DiGraph)
        # Invert connections, because <TF:is transcribed by:Module>
        original_connections = original_connections.reverse()
        self.graph.add_edges_from(original_connections.edges,
                                  origin=self.id_of_transcription)

    def clean_up_network(self) -> None:
        """Remove all non-binding TFs and unused modules."""
        self.remove_bidirectional_edges()
        self.remove_non_binding_tfs()
        self.remove_unused_modules()

    def remove_non_binding_tfs(self) -> None:
        """Delete transcription factors from the graph that do not bind to
        any module. I.e. that have an out_degree of 0.
        """
        # Find nodes which are tf and have out degree 0
        non_binding_tfs = []
        for tf_name in self.get_tfs():
            out_degree = self.graph.out_degree(tf_name)
            if out_degree == 0:
                non_binding_tfs.append(tf_name)
        # Remove them
        self.graph.remove_nodes_from(non_binding_tfs)

    def remove_unused_modules(self) -> None:
        """Delete modules which are unused. I.e. modules that are not regulated
        by any transcription factor, and also don't encode for any
        transcription factors.
        """
        unused_modules = []
        for module in self.get_modules():
            if (self.graph.in_degree(module) == self.graph.out_degree(module) == 0):
                unused_modules.append(module)
        # Remove them
        self.graph.remove_nodes_from(unused_modules)

    def remove_bidirectional_edges(self):
        """In case a module encodes a TF, ensure it never also shows that the
        TF bind to that module, because that is kinda senseless and probably
        a false-positive.
        """
        bidirectional_edges = [(u, v, data) for (u, v, data) in self.graph.edges(data=True)
                               if u in self.graph[v]]

        # Remove binds-to edges
        edges_to_be_removed = [(u, v) for (u, v, data) in bidirectional_edges
                               if data['origin'] == self.id_of_binding]

        self.graph.remove_edges_from(edges_to_be_removed)

    def get_module_module_network(self) -> ModuleRegulatoryNetwork:
        """Cut out all TFs and show direct relations between modules"""
        # Add edges
        edges_to_add = []
        for tf in self.get_tfs():
            original_module = list(self.graph.predecessors(tf))
            assert len(original_module) == 1, 'TF can only be transcribed by one module'
            # List contains only one item, extract it.
            original_module = original_module[0]
            target_modules = list(self.graph.successors(tf))
            # Potentially can make this add connections from multiple modules. But not implemented now.
            new_edges = [(original_module, target_module) for target_module in target_modules]
            edges_to_add.extend(new_edges)
        out_graph = nx.DiGraph()
        out_graph.add_edges_from(edges_to_add, origin=self.id_of_regulation)
        return ModuleRegulatoryNetwork(out_graph)

    def get_modules(self) -> list:
        """Return list of all module nodes"""
        return [node for node in self.graph.nodes if self.module_prefix in node]

    def get_tfs(self) -> list:
        """Return list of all transcription factor nodes"""
        return [node for node in self.graph.nodes if self.tf_prefix in node]

    def convert_to_ode(self) -> OdeInference:
        """Convert the graph to an object that contains all equations
        to perform fitting of the ODEs. This can only be called after .get_module_module_network() has been executed.
        """
        # Ensure that network is ModuleModule network
        assert all(self.id_of_regulation in edge for edge in self.graph.edges(data='origin')), 'Make sure you have removed all TFs from regulatory network and converted it to Module-Module network'
        # Todo keep names of edges involved here in some way?
        ode_out = OdeInference()
        ode_out.construct_formula_per_module(self.graph)
        return ode_out
