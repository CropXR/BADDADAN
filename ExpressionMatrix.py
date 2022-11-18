"""
Contains classes that contain gene expression levels, extracted from GEO.
"""


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster


class ExpressionMatrix:
    def __init__(self, df: pd.DataFrame):
        """
        :param df: Dataframe which contains gene expressions for various
                   biological samples.
        """
        self.df = df

    def __repr__(self):
        return (f'ExpressionMatrix with {len(self.df)} genes'
                f' and columns {self.df.columns.to_list()}')

    def get_column_names(self):
        return self.df.columns.to_numpy()

    def remove_non_wt(self):
        """Return ExpressionMatrix with only columns that originate from wild type"""
        col_mask = [col for col in self.df.columns if 'WT' in col]
        return ExpressionMatrix(self.df[col_mask])

    def plot_per_gene_std(self):
        """Plot per gene standard deviation across samples"""
        sns.histplot(self.df.std(axis=1))
        plt.show()

    def extract_train_test(self, train_cols: np.array_str,
                           test_cols: np.array_str):
        """Given a list of columns to use as train and test,
        return a ExpressionMatrixTrain, and ExpressionMatrixTest which
        are subsets of the ExpressionMatrix

        :param train_cols: Columns to use as training data
        :param test_cols: Columns to use as test data
        """
        assert not np.isin(test_cols, train_cols).any(), "Train and test columns overlap"
        expr_mat_train = ExpressionMatrixTraining(self.df[train_cols])
        expr_mat_test = ExpressionMatrixTest(self.df[test_cols])
        return expr_mat_train, expr_mat_test


class ExpressionMatrixTraining(ExpressionMatrix):
    def get_only_de_genes(self, std_cutoff: float = 1.0):
        """Return ExpressionMatrixTraining with only differentially
         expressed (de) genes.

        :param std_cutoff: Minimum standard deviation between samples
                           for a gene to be included. Default: 1.
        """
        return ExpressionMatrixTraining(self.df[self.df.std(axis=1) > std_cutoff])

    def extract_modules(self, n_cluster: int, inplace: bool = True):
        """Get mean expression per gene module based on
        clustering of expression correlation.

        :param n_cluster: Number of clusters
        :param inplace: If the cluster_id column should be added to self.df
                        in place. Default: True.
        """
        clustered_df = self._assign_clusters(n_cluster, inplace=inplace)

        molten_df = pd.melt(clustered_df, id_vars='cluster_id',
                            value_vars=clustered_df.columns[:-1],
                            ignore_index=False, var_name='sample',
                            value_name='expression')
        summary_df = molten_df.groupby(['sample', 'cluster_id']).mean().reset_index()
        return summary_df.pivot(index='sample', columns='cluster_id')

    def get_cluster_per_gene(self):
        """For each cluster, get Gene IDs. Can only be called after
        self.assign_clusters() has been called
        """
        assert 'cluster_id' in self.df.columns, 'Run assign_clusters() first!'
        return self.df.cluster_id.to_dict()

    def _assign_clusters(self, n_cluster: int, inplace: bool = True,
                         do_plotting: bool = False):
        """Hierarchically cluster genes based on correlation of expression,
        and extract given number of clusters.

        :param n_cluster: Number of clusters to extract
        :param inplace: If true, add cluster_id column to self.df, which describes
                        labels each gene with its corresponding cluster.
                        Default: True.
        :param do_plotting: If true, plot a clustermap with labels to
                            indicate the clustering
        :returns: Dataframe with cluster_id column which indicates the
                  clustering.
        """
        # Calculate pearson correlation between genes as distance measure
        subset_corr = self.df.T.corr()
        # Create linkage matrix and infer clusters
        linkage_matrix = linkage(subset_corr, method='complete')
        clustering = fcluster(linkage_matrix, n_cluster, 'maxclust')
        if do_plotting:
            # Create colours to use in clustermap
            lut = dict(zip([i for i in range(1, n_cluster + 1)],
                           sns.color_palette(n_colors=n_cluster)))
            row_colors = [lut[i] for i in clustering]
            sns.clustermap(subset_corr, row_linkage=linkage_matrix,
                           col_linkage=linkage_matrix, row_colors=row_colors)
            plt.show()
        if inplace:
            self.df = self.df.assign(cluster_id=clustering)
        return self.df.assign(cluster_id=clustering)


class ExpressionMatrixTest(ExpressionMatrix):
    def expressions_of_predefined_clusters(self,
                                           gene_to_cluster: dict):
        """From dict that maps genes to cluster, get mean expression per
        cluster.
        """
        assigned_modules = self.df.index.map(lambda x: gene_to_cluster.get(x))
        inference_df = self.df.assign(cluster_id=assigned_modules)
        molten_df = pd.melt(inference_df, id_vars='cluster_id',
                            value_vars=inference_df.columns[:-1],
                            ignore_index=False, var_name='sample',
                            value_name='expression')
        summary_df = molten_df.groupby(
            ['sample', 'cluster_id']).mean().reset_index()
        return summary_df.pivot(index='sample', columns='cluster_id')
