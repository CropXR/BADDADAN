"""
Some code written to do the analysis for the course
'Algorithms for Biomolecular Networks'
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def afbn_wrapper():
    """Does full analysis"""
    in_path = Path('data/afbn_exercise/cytoscape_cluster_outputs/cluster_table.csv')
    out_base_path = Path('data/afbn_exercise')
    generate_tf2network_input_per_clustering(in_path, out_base_path)
    compare_clustering(out_base_path)


def generate_tf2network_input_per_clustering(in_path: Path, out_base_path: Path):
    """Takes the full dataframe that is output from cytoscape and parses
     it into individual files that can be put into TF2Network
     """
    cluster_to_folder = {'__glayCluster': 'glay_clustering',
                         '__mclCluster2': 'markov_clustering',
                         '__mcodeCluster': 'mcode_clustering',
                         'hc_cluster': 'hierarchical_clustering'
                         }
    df = pd.read_csv(in_path)
    # Export files so they can be used on TF2Network
    df = df.drop(['selected', 'shared name'], axis=1)
    print()
    for col_name in df.columns[:4]:
        prefix = cluster_to_folder[col_name]
        out_path = out_base_path / prefix / f'{prefix}_tf2input.csv'
        selected_df = df[[col_name, 'name']]
        logging.info(f'Saving {out_path}')
        selected_df.to_csv(out_path, sep=' ', index=False, header=False)

    # Check pairwise agreement between clusterings?

def compare_clustering(in_dir: Path):
    df_list = []
    for dir in in_dir.glob('*_clustering'):
        tf2_output = list(dir.glob('*tf2network_output.tsv'))
        assert len(tf2_output) == 1
        tf2_output = tf2_output[0]
        df = parse_single_tf2_output(tf2_output)
        # Get sizes of clusters
        original_clustering = list(dir.glob('*tf2input.csv'))
        assert len(original_clustering) == 1
        original_clustering = original_clustering[0]
        cluster_df = pd.read_csv(original_clustering, sep=' ', names=['cluster','gene'])

        module_size_dict = cluster_df['cluster'].value_counts().to_dict()
        df['module_size'] = df['GeneSet'].map(lambda x: module_size_dict[int(float(x))])
        df_list.append(df)
    master_df = pd.concat(df_list, axis=0, ignore_index=True)
    master_df['module_size'] = master_df['module_size'].astype(float)
    sns.stripplot(data=master_df, x='method', y='-log10 q', hue='GeneSet', legend=False, size='module_size')
    plt.tight_layout()
    plt.show()

    # Show how many genes are included in the clusters?

    print()


def parse_single_tf2_output(some_tf2_output_path):
    # some_tf2_output_path = Path('data/afbn_exercise/glay_clustering/glay_cluster_tf2network_output.tsv')
    df = pd.read_csv(some_tf2_output_path, sep='\t')
    df['GeneSet'] = df['GeneSet'].astype(str)
    df = df[df['GeneSet'] != 'unnamed_set']
    df['-log10 q'] = -np.log10(df['q-value'])
    df['method'] = some_tf2_output_path.parts[-2]
    return df