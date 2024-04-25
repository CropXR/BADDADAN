from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Literal

import networkx as nx
import numpy as np
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
    tf_prefix = 'TF_molecule_'
    module_prefix = 'MODULE'

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def plot_network(self, draw_func: Callable = nx.draw,
                     out_path: Path|None = None, with_labels=True):
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
    def from_tf2_tsv(cls, in_path: Path, nr_top_hits: int | None = None,
                     q_value_cutoff: float | None = None) -> ModuleRegulatoryNetwork:
        """Create object from output of TF2Network file

        :param in_path: Path to .tsv output file of tf2network
        :param nr_top_hits: Keep certain number of top-scoring PWMs (position
                             weight matrices). Can be used as a quality cutoff.
        :param q_value_cutoff: Only keep edges with q-value below this cutoff.
                               Use either this or nr_top_hits, cannot both
                               be used at the same time.
        :return: Moduleregulatory network that contains TFs and modules
        """
        df = pd.read_csv(in_path, sep='\t')
        df = df[df['GeneSet'] != 'unnamed_set']
        # TODO at some point do this as node attribute instead of prepending to string
        df['target'] = cls.module_prefix + df['GeneSet'].astype(str)
        df['regulator_with_prefix'] = cls.tf_prefix + df['Regulator'].astype(str)
        df['origin'] = EdgeRelation.BINDS_TO

        if nr_top_hits and q_value_cutoff:
            raise ValueError('Cannot do both nr_top_hits and q_value_cutoff.'
                             ' Pick one or the other.')
        if nr_top_hits is not None:
            # Get only a certain number of top PWM hits for each module
            df = df.sort_values('rank').groupby('GeneSet').head(nr_top_hits)
        elif q_value_cutoff is not None:
            df = df[df['q-value'] < q_value_cutoff]

        my_graph = nx.from_pandas_edgelist(df, source='regulator_with_prefix',
                                           target='target',
                                           edge_attr='origin',
                                           create_using=nx.DiGraph)
        return cls(my_graph)

    def add_tf_module_mappings(self, path_to_orignal_cluster: Path,
                               from_tf2_input: bool = False) -> None:
        """Include what TFs are transcribed by which cluster.
        Add these edges with this method, input file is created from
        ExpressionMatrix object.
        """
        if from_tf2_input:
            # Needs some preprocessing to match up with original import format
            edges = []
            for line in path_to_orignal_cluster.open('r').readlines():
                try:
                    module, gene = line.strip().split()
                except ValueError as e:
                    logging.debug(f'Skipping {line} because {e}')
                    continue
                module = self.module_prefix + module
                gene = self.tf_prefix + gene.upper()
                edges.append((gene, module))
            original_connections = nx.DiGraph()
            original_connections.add_edges_from(edges)
        else:
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

    def get_module_module_network(self,
                                  tf_can_be_from_multiple_modules: bool = False
                                  ) -> ModuleRegulatoryNetwork:
        """Cut out all TFs and show direct relations between modules

        :param tf_can_be_from_multiple_modules: If true, a transcription factor
            can be transcribed my multiple modules. Default: false.
        :return: Module regulatory network that connects modules to modules.
        """
        # Find all potential edges and filter them to make sure they all agree
        candidate_edges = {}
        for tf in self.get_tfs():
            original_modules = list(self.graph.predecessors(tf))
            target_modules = list(self.graph.successors(tf))
            if not tf_can_be_from_multiple_modules:
                assert len(original_modules) == 1, f'TF ({tf}) can only be transcribed by one module'
            for original_module in original_modules:
                for target_module in target_modules:
                    regulation_type = self.graph.edges[tf, target_module]['origin']
                    if regulation_type == EdgeRelation.UP_OR_DOWN:
                        # Unclear regulations can be ignored
                        continue
                    # If edge not already in candidate edges dict, add it
                    if not (original_module, target_module) in candidate_edges:
                        candidate_edges[(original_module, target_module)] = {'origin': regulation_type,
                        'tf_name': [tf]}
                    # Check if edge agrees with existing edge
                    elif candidate_edges[(original_module, target_module)]['origin'] == regulation_type:
                        logging.debug(f'Found agreement between regulatory '
                                      f'direction of {original_module} -> {target_module}')
                        # Also add additional transcription factors
                        candidate_edges[(original_module, target_module)]['tf_name'].append(tf)
                        continue
                    else:
                        raise ValueError('Disagreement between regulatory directions'
                                         f' of {original_module} -> {target_module}')

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
                                       corr_cutoff: float = 0.3,
                                       do_plotting: bool = False,
                                       remove_low_corr: bool = False,
                                       assert_correlated: bool = True):
        """Check if TFs are 'created' by their module.

        To do this, we verify that the mean expression of the module is
        positively correlated with the transcription factor it produces.
        """

        tfs_to_remove = set()

        debug_corrs = []
        all_module_tf_pairs = self.get_filtered_module_tf_edges(
            EdgeRelation.TRANSCRIBED_BY)
        for module_name, tf_name in all_module_tf_pairs:
            # Remove the prefixes from module and tf
            module_without_prefix = module_name.removeprefix(self.module_prefix)
            module_without_prefix = int(float(module_without_prefix))
            tf_name_without_prefix = tf_name.removeprefix(self.tf_prefix)
            corr = expressions.get_correlation(module_without_prefix,
                                               tf_name_without_prefix,
                                               plot=False, method='pearson')
            debug_corrs.append(corr)
            if not remove_low_corr and assert_correlated:
                assert corr > corr_cutoff, \
                    (f'HUH?! Module {module_name} is not positively correlated '
                     f'with the TF ({tf_name}) it produces. Can be fixed by '
                     f'running this function with remove_low_corr=True '
                     f'or assert_correlated=False')
            elif corr < corr_cutoff:
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
        return debug_corrs


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
            if module_without_prefix == 'unnamed_set':
                logging.warning(f'Skipping {module_name} because it is not a number')
                continue
            module_without_prefix = int(float(module_without_prefix))
            tf_name_without_prefix = tf_name.removeprefix(self.tf_prefix)
            # See how the tf and module correlate
            corr = expressions.get_correlation(module_without_prefix,
                                               tf_name_without_prefix,
                                               plot=False)
            debug_corrs.append(corr)

            if corr < -threshold:
                direction = EdgeRelation.DOWNREGULATES
            elif corr > threshold:
                direction = EdgeRelation.UPREGULATES
            else:
                direction = EdgeRelation.UP_OR_DOWN
            edge_attr_dict[(tf_name, module_name)] = {'origin': direction,
                                                      'cor_strength': corr}
        nx.set_edge_attributes(self.graph, edge_attr_dict)
        if do_plotting:
            sns.set_style()
            sns.swarmplot(debug_corrs)
            plt.show()

    def annotate_module_members(
            self, expression_matrix: ExpressionMatrixTimeSeries):
        """For each module, assign an attribute that describes all
        the genes that the module contains"""
        cluster_to_gene_dict = expression_matrix.get_genes_per_cluster()
        module_name_as_key = {f'{self.module_prefix}{key}': value
                              for key, value in cluster_to_gene_dict.items()}
        nx.set_node_attributes(self.graph, module_name_as_key, name='gene_names')

    def save_for_cytoscape(self, out_path: Path):
        """Save edge list for opening network in Cytoscape

        :param out_path: tsv file to save file to
        """
        # TODO correctly add edge attributes to this output
        # nx.write_edgelist(self.graph, out_path)
        # Save gene pairs to a file
        with out_path.open('w+') as f:
            f.write("Gene1\tGene2\tCorr_strength\n")
            for pair in self.graph.edges(data=True):
                f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\n")

    def keep_only_modules_of_interest(self, expr_mat: ExpressionMatrixTimeSeries):
        """Only maintain nodes that belong to a module that is still in the expr_mat

        Used if you have filtered certain modules in the expression data, and
        only want to see what their intermodular network looks like.

        :param expr_mat: Expression matrix that only contains clusters of interest
        """
        valid_edges = []
        modules = expr_mat.get_genes_per_cluster().keys()
        for edge in self.graph.edges(data=True):
            edge_type = edge[2]['origin']
            if edge_type == EdgeRelation.BINDS_TO:
                module_index = 1
            elif edge_type == EdgeRelation.TRANSCRIBED_BY:
                module_index = 0
            else:
                raise NotImplementedError('Currently can only keep modules '
                                          'when selecting in a TF->MODULE '
                                          'network')

            module_int = int(edge[module_index].replace(self.module_prefix, ''))
            if module_int in modules:
                valid_edges.append(edge)

        new_graph = nx.from_edgelist(valid_edges, nx.DiGraph)
        self.graph = new_graph

    def get_intermodular_connection_df(self) -> pd.DataFrame:
        """Return a dataframe that contains one row for each TF

        Each row also contains the modules that it connects
        (i.e. module in which it is produced, and module to
        which it binds.
        Finally, also return the strength of the correlation
        between the TF and the module to which it binds.
        """
        out_list = []
        for tf in self.get_tfs():
            original_modules = list(self.graph.predecessors(tf))
            assert len(original_modules) == 1
            from_module = original_modules[0]
            target_modules = list(self.graph.successors(tf))
            for target_module in target_modules:
                corr = self.graph.edges[tf, target_module]['cor_strength']
                out_entry = [from_module, target_module, corr, tf]
                out_list.append(out_entry)
        df = pd.DataFrame.from_records(out_list, columns=['from', 'to', 'cor', 'tf'])
        return df

    def check_consistency_between_module_regulations(self) -> pd.Series:
        """If more than 1 TF connects modules, see if they all agree (e.g. all up or downregulate)

        To do this check if all connections are either above or below a various correlation thresholds, and return
        the fraction of all connections in which the TFs agree.
        """
        df = self.get_intermodular_connection_df()
        grouped_df = df.groupby(['from', 'to'])
        at_least_double = grouped_df.filter(lambda x: len(x) > 1)
        at_least_double_grpd = at_least_double.groupby(['from', 'to'])
        # Get if they agree?
        def check_agreement(x):
            out = []
            my_range = np.arange(0,1,0.1)
            for cor_thresh in my_range:
                is_consistent = all(x['cor'] > cor_thresh) or all(x['cor'] < cor_thresh)
                out.append(is_consistent)
            return pd.Series(out, index=my_range)
        agrees_df = at_least_double_grpd.apply(check_agreement)
        return agrees_df.mean()

    def see_how_many_tfs_between_modules(self) -> pd.DataFrame:
        """Get a distribution of how many TFs are between modules

        Used for checking if TFs are always in agreement with one another
        """

        df = self.get_intermodular_connection_df()
        grouped_df = df.groupby(['from', 'to'])
        size_distribution = grouped_df.size()
        size_distribution.name = 'nr_tfs_between_modules'
        return size_distribution.reset_index()


