"""
Some code written to do the analysis for the course
'Algorithms for Biomolecular Networks'
"""
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries


def afbn_wrapper():
    """Does full analysis"""
    out_base_path = Path('data/afbn_exercise/unweighted_edge_input')
    in_path = out_base_path / 'cytoscape_cluster_outputs/cluster_table.csv'
    expr_mat_path = Path(
        'data/time_series_datasets/tf2network_approach/2000_highest_mad_log2/GSE5628_family_ExpressionMatrixTime.pickle')

    generate_tf2network_input_per_clustering(in_path, out_base_path)
    compare_clustering(out_base_path)
    for directory in out_base_path.glob('*_clustering'):
        if 'hier' in directory.name:
            continue
        # logging.info('-'*15+'\n')
        graph_output_path = directory / 'module_graph.pkl'
        logging.info(f'Scanning {directory.name}')
        with expr_mat_path.open('rb') as f:
            expr_mat = pickle.load(f)
        from_tf2_output_to_network(next(directory.glob('*_tf2network_output.tsv')),
                                   next(directory.glob('*_tf2input.csv')),
                                   expr_mat,
                                   graph_output_path)

    records = []
    for graph_path in out_base_path.parent.rglob('module_graph.pkl'):
        parent_path = graph_path.parent
        method = parent_path.parts[-1].replace('_', ' ').capitalize()
        graph: nx.DiGraph = nx.read_gpickle(graph_path)
        modules = graph.nodes(data=True)
        nr_modules = len(modules)
        nr_genes = 0
        for module_name, data_dict in modules:
            nr_genes += len(data_dict['gene_names'])
            # Find how many genes in the module
        records.append((method, nr_modules, nr_genes))
    result_df = pd.DataFrame.from_records(records,
                                          columns=['method', 'nr_modules',
                                                   'nr_genes'])
    sns.set_theme()
    # print(result_df.method)
    ordaaah = ["Glay clustering",
               "Glay clustering 0.7 cutoff",
               "Hierarchical clustering",
               "Markov clustering unweighted edges",
               "Markov clustering weighted edges",
               "Mcode clustering"]
    ax = sns.scatterplot(data=result_df, y='nr_modules', x='nr_genes',
                         hue='method', hue_order=ordaaah)
    ax.set_xlim(0, 1750)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Number of genes')
    ax.set_ylabel('Number of modules')
    # ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def generate_tf2network_input_per_clustering(in_path: Path,
                                             out_base_path: Path):
    """Takes the full dataframe that is output from cytoscape and parses
     it into individual files that can be put into TF2Network
     """
    cluster_to_folder = {'__glayCluster': 'glay_clustering',
                         '__mclCluster2': 'markov_clustering',
                         '__mcodeCluster': 'mcode_clustering',
                         'hc_cluster': 'hierarchical_clustering',
                         'mcl_selected_nodes': 'mcl_distance_informed_clustering'}
    df = pd.read_csv(in_path)
    # Export files so they can be used on TF2Network
    df = df.drop(['selected', 'shared name'], axis=1)
    print()
    for col_name in df.columns[:4]:
        if col_name not in cluster_to_folder:
            logging.info(f'Skipping {col_name}')
            continue
        prefix = cluster_to_folder[col_name]
        out_path = out_base_path / prefix / f'{prefix}_tf2input.csv'
        out_path.parent.mkdir(exist_ok=True, parents=True)
        selected_df = df[[col_name, 'name']]
        logging.info(f'Saving {out_path}')
        selected_df.to_csv(out_path, sep=' ', index=False, header=False)

    # Check pairwise agreement between clusterings?


def compare_clustering(in_dir: Path):
    df_list = []
    for directory in in_dir.glob('*_clustering*'):
        tf2_output = list(directory.glob('*tf2network_output.tsv'))
        if not tf2_output:
            continue
        assert len(tf2_output) == 1
        tf2_output = tf2_output[0]
        df = parse_single_tf2_output(tf2_output)
        # Get sizes of clusters
        original_clustering = list(directory.glob('*tf2input.csv'))
        assert len(original_clustering) == 1
        original_clustering = original_clustering[0]
        cluster_df = pd.read_csv(original_clustering, sep=' ',
                                 names=['cluster', 'gene'])

        module_size_dict = cluster_df['cluster'].value_counts().to_dict()
        df['module_size'] = df['GeneSet'].map(
            lambda x: module_size_dict[int(float(x))])
        df_list.append(df)
    master_df = pd.concat(df_list, axis=0, ignore_index=True)
    master_df['module_size'] = master_df['module_size'].astype(float)

    # How many above 10
    master_df.groupby('method')['-log10 q'].apply(
        lambda x: (x > 10).sum()).reset_index()

    counts_per_module = master_df.groupby('method')['GeneSet'].value_counts()
    counts_per_module = counts_per_module.reset_index(name='Nr of tfs')
    sns.boxplot(data=counts_per_module, x='method', y='Nr of tfs')
    sns.stripplot(data=counts_per_module, x='method', y='Nr of tfs')
    plt.tight_layout()
    # sns.countplot(data=master_df, x='method', hue='GeneSet'); plt.legend().set_visible(False); plt.show()
    sns.stripplot(data=master_df, x='method', y='-log10 q',
                  hue='module_size')  # , legend=False, size='module_size')
    sns.violinplot(data=master_df, x='method',
                   y='-log10 q')  # , hue='GeneSet', legend=False, size='module_size')
    plt.tight_layout()
    plt.show()

    # Show how many genes are included in the clusters?

    print()


def from_tf2_output_to_network(tf2_output_file: Path,
                               tf2_input_file: Path,
                               expression_matrix: ExpressionMatrixTimeSeries,
                               out_path: Path):
    ## Now it's time to get the network that we need ##
    # This file has to be created manually from the TF2 website
    expression_matrix.assign_clusters_from_tf2_input(tf2_input_file)

    my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(
        tf2_output_file, nr_top_hits=None)
    my_grn.add_tf_module_mappings(tf2_input_file, from_tf2_input=True)
    my_grn.clean_up_network()
    # my_grn.plot_network(with_labels=True)
    my_grn.check_if_tfs_created_by_module(expression_matrix, do_plotting=False,
                                          remove_low_corr=True)
    my_grn.set_up_or_downregulation(expression_matrix, do_plotting=False,
                                    threshold=.3)
    # my_grn.plot_network(nx.draw_kamada_kawai, with_labels=False)
    module_module = my_grn.get_module_module_network()
    module_module.annotate_module_members(expression_matrix)
    module_module.plot_network()
    nx.write_gpickle(module_module.graph, out_path)


def parse_single_tf2_output(some_tf2_output_path):
    # some_tf2_output_path = Path('data/afbn_exercise/glay_clustering/glay_cluster_tf2network_output.tsv')
    df = pd.read_csv(some_tf2_output_path, sep='\t')
    df['GeneSet'] = df['GeneSet'].astype(str)
    df = df[df['GeneSet'] != 'unnamed_set']
    df['-log10 q'] = -np.log10(df['q-value'])
    df['method'] = some_tf2_output_path.parts[-2]
    return df
