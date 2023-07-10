from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Literal

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries


class EdgeRelation(Enum):
    """Describes what the relationship between nodes means in the graph.

    Relationship can be that a TF binds to a module, that a modules produces
    a TF, or that a module regulates another module.
    """
    BINDS_TO = 'binds_to'
    TRANSCRIBED_BY = 'transcribed_by'
    REGULATES = 'regulates'
    UPREGULATES = 'upregulates'
    DOWNREGULATES = 'downregulates'
    UP_OR_DOWN = 'unclear_regulation_direction'


class ModuleRegulatoryNetwork:
    # Standard prefixes
    tf_prefix = 'TF_'
    module_prefix = 'MODULE'

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def plot_network(self, draw_func: Callable = nx.draw,
                     out_path: Path = None, with_labels=True):
        """Plot network using matplotlib."""

        node_color_map = ['blue' if (node in self.get_tfs()) else 'orange'
                          for node in self.graph.nodes]

        edge_mapping_dict = {EdgeRelation.REGULATES: 'green',
                             EdgeRelation.BINDS_TO: 'brown',
                             EdgeRelation.TRANSCRIBED_BY: 'black',
                             EdgeRelation.UPREGULATES: 'green',
                             EdgeRelation.DOWNREGULATES: 'red',
                             EdgeRelation.UP_OR_DOWN: 'gray'}

        edge_color_map = [edge_mapping_dict[origin] for _, _, origin
                          in self.graph.edges.data('origin')]

        draw_func(self.graph, node_color=node_color_map,
                  edge_color=edge_color_map, with_labels=with_labels, arrowsize=30)
        if out_path:
            plt.savefig(out_path)
        else:
            plt.show()

    @classmethod
    def from_lpan_edge_csv(
            cls, lpan_file_path: Path, top_rank: int = None
    ) -> ModuleRegulatoryNetwork:
        """Constructor to create object from lpan edge csv.

        :param lpan_file_path: Path to csv output by LPAN.
        E.g. aracne_network_edges.csv
        :param top_rank: If provided, keep only the top_rank  number of
        connections. I.e. the strongest one predicted by lpan.
        """
        some_df = pd.read_csv(lpan_file_path)
        if top_rank:
            some_df = some_df[some_df['rank'] < top_rank]
        some_df['origin'] = EdgeRelation.BINDS_TO
        a_graph = nx.from_pandas_edgelist(some_df, source='regulator',
                                          target='target',
                                          edge_attr='origin',
                                          create_using=nx.DiGraph)
        return cls(a_graph)

    @classmethod
    def from_tf2_tsv(cls, in_path: Path, nr_top_hits: int | None = None) -> ModuleRegulatoryNetwork:
        """Create object from output of TF2Network file

        :param nr_top_hits: Keep certain number of top-scoring PWMs (position
                             weight matrices). Can be used as a quality cutoff.
        :param in_path: Path to .tsv output file of tf2network
        :return: Moduleregulatory network that contains TFs and modules
        """
        df = pd.read_csv(in_path, sep='\t')
        df['target'] = cls.module_prefix + df['GeneSet'].astype(str)
        df['regulator_with_prefix'] = cls.tf_prefix + df['Regulator'].astype(str)
        df['origin'] = EdgeRelation.BINDS_TO

        if nr_top_hits is not None:
            # Get only a certain number of top PWM hits for each module
            filtered_df = df.sort_values('rank').groupby('GeneSet').head(nr_top_hits)
        else:
            filtered_df = df
        my_graph = nx.from_pandas_edgelist(filtered_df, source='regulator_with_prefix',
                                           target='target',
                                           edge_attr='origin',
                                           create_using=nx.DiGraph)
        return cls(my_graph)

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
                                  origin=EdgeRelation.TRANSCRIBED_BY)

    def clean_up_network(self) -> None:
        """Remove all non-binding TFs and unused modules."""
        # TODO perhaps we can keep bidirectional edges, i.e. module
        #  self-regulation might happen(?)
        self.remove_bidirectional_edges()
        self.remove_non_binding_tfs()
        self.remove_unused_modules()
        self.remove_untranscribed_tfs()

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
        self.graph.remove_nodes_from(unused_modules)

    def remove_bidirectional_edges(self):
        """In case a module encodes a TF, ensure it never also shows that the
        TF bind to that module, because that is kinda senseless and probably
        a false-positive.
        """
        bidirectional_edges = [(u, v, data) for (u, v, data)
                               in self.graph.edges(data=True)
                               if u in self.graph[v]]

        # Remove binds-to edges
        edges_to_be_removed = [(u, v) for (u, v, data) in bidirectional_edges
                               if data['origin'] == EdgeRelation.BINDS_TO]

        self.graph.remove_edges_from(edges_to_be_removed)

    def get_module_module_network(self) -> ModuleRegulatoryNetwork:
        """Cut out all TFs and show direct relations between modules"""
        # Find all potential edges and filter them to make sure they all agree
        candidate_edges = {}
        for tf in self.get_tfs():
            original_module = list(self.graph.predecessors(tf))
            assert len(original_module) == 1, f'TF ({tf}) can only be transcribed by one module'
            # List contains only one item, extract it.
            original_module = original_module[0]
            target_modules = list(self.graph.successors(tf))
            for target_module in target_modules:
                regulation_type = self.graph.edges[tf, target_module]['origin']
                if regulation_type == EdgeRelation.UP_OR_DOWN:
                    # Unclear regulations can be ignored
                    continue
                # If edge not already in candidate edges dict, add it
                if not (original_module, target_module) in candidate_edges:
                    candidate_edges[(original_module, target_module)] = dict(origin=regulation_type)
                # Check if edge agrees with existing edge
                elif candidate_edges[(original_module, target_module)]['origin'] == regulation_type:
                    logging.debug(f'Found agreement between regulatory '
                                  f'direction of {original_module} -> {target_module}')
                    continue
                else:
                    raise ValueError('Disagreement between regulatory directions'
                                     ' of {original_module} -> {target_module}')

        edge_list = []
        for (origin, target), data_dict in candidate_edges.items():
            edge_list.append((origin, target, data_dict))

        out_graph = nx.DiGraph()
        out_graph.add_edges_from(edge_list)
        return ModuleRegulatoryNetwork(out_graph)

    def get_modules(self) -> list[str]:
        """Return list of all module nodes"""
        return [node for node in self.graph.nodes if self.module_prefix in node]

    def get_tfs(self) -> list[str]:
        """Return list of all transcription factor nodes"""
        return [node for node in self.graph.nodes if self.tf_prefix in node]

    def remove_untranscribed_tfs(self):
        """Remove TFs from graph that are not transcribed by any of the modules.
        I.e. TFs that have an in-degree of 0
        """
        non_binding_tfs = []
        for tf_name in self.get_tfs():
            out_degree = self.graph.in_degree(tf_name)
            if out_degree == 0:
                non_binding_tfs.append(tf_name)
        # Remove them
        self.graph.remove_nodes_from(non_binding_tfs)

    def get_filtered_module_tf_edges(
            self, edge_filter_id: EdgeRelation
    ) -> Generator[tuple[str, str]]:
        """Get pairs of module and transcription factor.

        Use edge_filter_id to filter the edges. If 'binds_to' is provided
        as filter, you will get pairs of modules and the TFs that bind to them.
        These can be used to infer if TFs up- or downregulate a module.

        If 'transcribed_by' is provided as a filter, you will get pairs of
        modules, and the transcripiton factor that they transcribe.
        These pairs can be used to verify that modules are positively
        correlated with the products they produce - (as you would expect).

        :returns: Generator of tuples. If filtering for transcription,
        the first item in the tuple is the module, the second is the TFs.
        If filtering for binding, the first item in the tuple is the TF,
        and the second is the module.
        """
        for node1, node2, origin in self.graph.edges(data='origin'):
            if origin == edge_filter_id:
                yield node1, node2

    def check_if_tfs_created_by_module(self,
                                       expressions: ExpressionMatrixTimeSeries,
                                       do_plotting: bool = False,
                                       remove_low_corr: bool = False):
        """Check if TFs are 'created' by their module.

        To do this, we verify that the mean expression of the module is
        positively correlated with the transcription factor it produces.
        """
        if remove_low_corr:
            tfs_to_remove = set()

        debug_corrs = []
        all_module_tf_pairs = self.get_filtered_module_tf_edges(
            EdgeRelation.TRANSCRIBED_BY)
        for module_name, tf_name in all_module_tf_pairs:
            # Remove the prefixes from module and tf
            module_without_prefix = module_name.removeprefix(self.module_prefix)
            module_without_prefix = int(module_without_prefix)
            tf_name_without_prefix = tf_name.removeprefix(self.tf_prefix)
            corr = expressions.get_correlation(module_without_prefix,
                                               tf_name_without_prefix,
                                               plot=False, method='pearson')
            debug_corrs.append(corr)
            if not remove_low_corr:
                assert corr > .3, f'HUH?! Module {module_name} is not positively correlated ' \
                                  f'with the TF ({tf_name}) it produces. Can be fixed by running this function with remove_low_corr=True'
            elif corr < .3:
                tfs_to_remove.add(tf_name)
            else:
                continue
        if remove_low_corr and tfs_to_remove:
            logging.warning('REMOVING LOW CORRELATION TFS BECAUSE IT '
                            'WAS SPECIFIED IN THE FUNCTION CALL')
            logging.info(
                f'Removing {tfs_to_remove} because correlation between their '
                f'expression and the module that produces them is too low'
            )
            self.graph.remove_nodes_from(tfs_to_remove)

        if do_plotting:
            sns.set_style()
            sns.boxplot(debug_corrs)
            sns.swarmplot(debug_corrs, color=sns.color_palette()[1])
            plt.ylim((-.5, 1))
            plt.show()
        return True

    def set_up_or_downregulation(self,
                                 expressions: ExpressionMatrixTimeSeries,
                                 threshold: float = .2,
                                 do_plotting=False):
        """For each TF, find it out if up-/downregulates its target module.

        This is done by looking at the correlation between the TF and the target
        module. If high correlation, we assign upregulation. If low correlation,
        we assign downregulation.

        :param expressions: ExpressionMatrix which contains expressions of genes
        :param threshold: When to assign up/down regulation. If below threshold,
        connection is assigned the UP_OR_DOWN label.
        """
        edge_attr_dict = {}
        debug_corrs = []
        all_tfs_and_binding_sites = self.get_filtered_module_tf_edges(
            edge_filter_id=EdgeRelation.BINDS_TO)
        for tf_name, module_name in all_tfs_and_binding_sites:
            # Remove the module and tf prefix
            module_without_prefix = module_name.removeprefix(
                self.module_prefix)
            module_without_prefix = int(module_without_prefix)
            tf_name_without_prefix = tf_name.removeprefix(self.tf_prefix)
            # See how the tf and module correlate
            corr = expressions.get_correlation(module_without_prefix,
                                               tf_name_without_prefix, plot=False)
            debug_corrs.append(corr)
            if corr < -threshold:
                direction = EdgeRelation.DOWNREGULATES
            elif corr > threshold:
                direction = EdgeRelation.UPREGULATES
            else:
                direction = EdgeRelation.UP_OR_DOWN
            edge_attr_dict[(tf_name, module_name)] = {'origin': direction}
        nx.set_edge_attributes(self.graph, edge_attr_dict)
        if do_plotting:
            sns.set_style()
            sns.swarmplot(debug_corrs)
            plt.show()
