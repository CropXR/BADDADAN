"""
Contains classes that contain matrices of gene expression levels.
"""
from __future__ import annotations
import logging
from pathlib import Path


import numpy as np
import pandas as pd
import seaborn as sns
import GEOparse
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import qnorm

from Expressions.ExpressionArrayAnnotation import ExpressionArrayAnnotation
from helpers import get_info_from_gse5628


class ExpressionMatrix:
    # Default prefixes to use when exporting to LPAN rscript files
    tf_prefix = 'TF_'
    module_prefix = 'MODULE'

    def __init__(self, df: pd.DataFrame):
        """Create expression matrix directly from Pandas dataframe

        :param df: Dataframe which contains gene expressions for various
                   biological samples
        """
        self.df = df
        self.has_been_clustered = False

    def __repr__(self):
        return (f'ExpressionMatrix with {len(self.df)} genes'
                f' and columns {self.df.columns.to_list()}')

    @classmethod
    def from_csv(cls, file_path: Path,
                 array_annotation: ExpressionArrayAnnotation = None,
                 log2_transform: bool = False,
                 sep: str = '\t'):
        df = pd.read_csv(file_path, sep=sep, index_col=0)
        df = cls._df_preprocessing(array_annotation, df, log2_transform)
        return cls(df)

    @classmethod
    def from_geo_file(cls, file_path: Path,
                      array_annotation: ExpressionArrayAnnotation = None,
                      log2_transform: bool = False):
        """From a file path, correctly parse GEO expression file and
        return ExpressionMatrix object.

        :param file_path: path to GEO expression file. Works on .soft format,
                      others have not been tested.
        :param array_annotation: Object which maps probe names of AGI names.
                                 Should have probe_to_agi() method.
        :param log2_transform: Log2 transform the expression data.
        """
        gse = GEOparse.get_GEO(filepath=str(file_path), silent=True)
        # Merge all samples into one dataframe
        df = gse.pivot_samples("VALUE")
        logging.info('Dropping AFFX probes, because they are control')
        df = df.loc[df.index.map(lambda x: 'AFFX' not in x), :]
        df = cls._df_preprocessing(array_annotation, df, log2_transform)
        # Convert sample names to titles that humans can understand
        better_name_dict = gse.phenotype_data.title.to_dict()
        df.columns = [better_name_dict[old_col] for old_col in df.columns]
        return cls(df)

    @classmethod
    def _df_preprocessing(cls, array_annotation: ExpressionArrayAnnotation,
                          df: pd.DataFrame, log2_transform: bool):
        if array_annotation:
            # Get gene names based on ExpressionAnnotation object
            logging.info('Converting probe names to genes...')
            new_indices = df.index.map(array_annotation.probe_to_agi)
            # Count how many probe names were not mapped to a gene by the
            # annotation file, i.e. their name did not change.
            unmapped_probes = new_indices.intersection(df.index)

            logging.info(
                f'Could not find annotation of {len(unmapped_probes)} probes '
                f'({(len(unmapped_probes) / len(df.index)) * 100:.2f}%). '
                f'Proceeding with their original names')
            if len(unmapped_probes) < 10:
                for probe in unmapped_probes:
                    logging.info(probe)
            df.index = new_indices
        if log2_transform:
            df = np.log2(df)
        return df

    def get_sample_names(self) -> np.array:
        """Returns all names of samples"""
        return self.df.columns.to_numpy()

    def get_gene_names(self) -> list:
        """Returns names of all gene names"""
        return self.df.index.to_list()

    def get_only_wt_samples(self) -> ExpressionMatrix:
        """Return ExpressionMatrix with only columns that originate from wild type"""
        col_mask = [col for col in self.df.columns if 'WT' in col]
        return ExpressionMatrix(self.df[col_mask])

    def plot_per_gene_std(self) -> None:
        """Plot per gene standard deviation across samples"""
        sns.histplot(self.df.std(axis=1))
        plt.show()

    def plot_sample_gene_heatmap(self) -> None:
        sns.clustermap(self.df, method='complete', metric='correlation',
                       yticklabels=False)
        plt.show()

    def extract_train_test(self, train_cols: np.array_str,
                           test_cols: np.array_str) -> tuple[ExpressionMatrixTraining, ExpressionMatrixTest]:
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

    def keep_only_de_genes(self, std_cutoff: float = 1.0) -> None:
        """Remove non-differentially expressed (de) genes.

        :param std_cutoff: Minimum standard deviation between samples
                           for a gene to be included. Default: 1.0
        """
        de_genes = self.df[self.df.std(axis=1) > std_cutoff]
        self.df = de_genes

    def quantile_normalize(self, ref_mappings: ExpressionMatrix | None = None):
        if ref_mappings is None:
            # For train dataset just do complete normalisation
            self.df = qnorm.quantile_normalize(self.df)
        else:
            reference_df = ref_mappings.df.copy()
            # remove duplicates
            reference_df = reference_df[~reference_df.index.duplicated(keep='first')]

            mutual_genes = set(self.df.index) & set(reference_df.index)
            self.df = self.df[self.df.index.isin(mutual_genes)]
            # Get mean values for overlapping genes, and use these to infer values
            reference_df = reference_df[reference_df.index.isin(mutual_genes)]

            my_mappings = reference_df.loc[self.df.index].mean(axis=1)
            # For test dataset map onto mean values that were calculated from train dataset
            self.df = qnorm.quantile_normalize(self.df, target=my_mappings)

    def to_expressionmatrix_training(self):
        assert type(self) != ExpressionMatrixTraining, 'Is already an ExpressionMatrixTraining object. Conversion is pointless.'
        return ExpressionMatrixTraining(self.df)

    def to_expressionmatrix_test(self):
        assert type(self) != ExpressionMatrixTest, 'Is already an ExpressionMatrixTest object. Conversion is pointless.'
        return ExpressionMatrixTest(self.df)



class ExpressionMatrixTraining(ExpressionMatrix):
    """Can be created from ExpressionMatrix by command like:

    my_expression_matrix = ExpressionMatrix.from_geo(some_path)
    training_df = ExpressionMatrixTraining(my_expression_matrix.df)

    """
    def extract_module_expressions(self, n_cluster: int,
                                   for_static_predictions = False,
                                   do_plotting: bool = False) -> pd.DataFrame:
        """Get mean expression per gene module based on
        clustering of expression correlation. And return as dataframe.

        :param n_cluster: Number of clusters
        """
        # Make sure clustering has been performed
        if not self.has_been_clustered:
            logging.info('Not clustered yet, performing clustering now')
            self.do_hierachical_clustering(n_cluster, do_plotting=do_plotting)
        else:
            logging.info('Already clustered, will not perform clustering again')

        molten_df = pd.melt(self.df, id_vars='cluster_id',
                            value_vars=self.df.columns[:-1],
                            ignore_index=False, var_name='sample',
                            value_name='expression')
        summary_df = molten_df.groupby(['sample', 'cluster_id']).mean().reset_index()
        if for_static_predictions:
            return summary_df.pivot(index='sample', columns='cluster_id', values='expression')
        return summary_df

    def save_cluster_gene_edge_list(self, out_file_path: Path,
                                    tf_filter_list: None | list[str] = None):
        """Save edge list that maps transcription factors to their gene"""
        if tf_filter_list is None:
            # Is this realistically ever used without tf_filter_
            out_df = self.df
        else:
            prefixed_index = self.tf_prefix + self.df.index.astype(str)
            out_df = self.df[prefixed_index.isin(tf_filter_list)]
            out_df.index = prefixed_index[prefixed_index.isin(tf_filter_list)]
            out_df['cluster_id'] = self.module_prefix + out_df.cluster_id.astype(str)
        out_df.to_csv(out_file_path, sep=' ', columns=['cluster_id'],
                      header=False)

    def get_cluster_per_gene(self) -> dict:
        """For each gene, get its cluster_id Can only be called after
        self.do_hierachical_clustering() has been called.

        :returns: Dict with gene name as key and cluster_ID as value .
        """
        assert 'cluster_id' in self.df.columns,\
                'Run do_hierachical_clustering() first!'
        return self.df.cluster_id.to_dict()

    def get_genes_per_cluster(self) -> dict:
        """For each cluster, get Gene IDs. Can only be called after
        self.do_hierachical_clustering() has been called.

        :returns: Dict with keys cluster_id and values a list of all genes in that cluster .
        """
        assert 'cluster_id' in self.df.columns,\
                'Run do_hierachical_clustering() first!'

        return self.df.groupby('cluster_id').groups

    def do_hierachical_clustering(self, n_cluster: int,
                                  do_plotting: bool = False) -> None:
        """Hierarchically cluster genes based on correlation of expression,
        and extract given number of clusters. Adds column cluster_id to
        object, which indicates for each gene to which cluster it belongs.

        :param n_cluster: Number of clusters to extract
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
        self.df = self.df.assign(cluster_id=clustering)
        self.has_been_clustered = True


class ExpressionMatrixTest(ExpressionMatrix):
    def expressions_of_predefined_clusters(self,
                                           gene_to_cluster: dict) -> pd.DataFrame:
        """From dict that maps genes to cluster, get mean expression per
        cluster.

        :param gene_to_cluster: Dictionary that maps gene names to cluster ids
        :return: Mean expression per cluster
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

    def expressions_of_predefined_clusters_seed_data(self,
                                           gene_to_cluster: dict) -> pd.DataFrame:
        """From dict that maps genes to cluster, get mean expression per
        cluster. Use this implementation when dealing with the seed dataset

        :param gene_to_cluster: Dictionary that maps gene names to cluster ids
        :return: Mean expression per cluster
        """
        assigned_modules = self.df.index.map(lambda x: gene_to_cluster.get(x))
        inference_df = self.df.assign(cluster_id=assigned_modules)
        return inference_df.groupby('cluster_id').mean().T


class ExpressionMatrixTimeSeries(ExpressionMatrixTraining):
    def keep_only_shoot(self) -> None:
        """Keep only columns that originate from shoot"""
        col_mask = [col for col in self.df.columns if 'Shoot' in col]
        self.df = self.df[col_mask]

    def plot_clusters_over_time(self) -> None:
        """Plot mean expression of clusters over time.

        """
        sns.set_theme()
        some_df = self._get_gene_expression_long_form()
        # sns.lineplot(data=some_df, x='time', y='expression',
        #              hue='cluster', palette=sns.color_palette())

        sns.lineplot(data=some_df, x='time', y='expression',
                     hue='cluster', style='replicate', palette=sns.color_palette(), errorbar='sd')
        plt.show()
        # sns.stripplot(data=some_df, x='time', y='expression',
        #                hue='cluster', palette=sns.color_palette())
        # ax = sns.violinplot(data=some_df, x='time', y='expression',
        #                hue='cluster', palette=sns.color_palette()) #, dodge=False, inner=None)
        # plt.setp(ax.collections, alpha=.5)
        # sns.lineplot(data=some_df, x='elapsed_mins', y='expression',
        #              hue='cluster_id', style='tissue', palette=sns.color_palette())
        # plt.show()

    def _get_cluster_expression_long_form(self, n_clusters):
        """Get dataframe which shows expression of clusters over time
        in long-form dataframe
        """
        out_df = self.extract_module_expressions(n_clusters, do_plotting=False)
        # Get time point info from sample names
        new_cols = get_info_from_gse5628(out_df['sample'].to_list())
        out_df = pd.concat([out_df, pd.DataFrame.from_dict(new_cols)], axis=1)
        out_df['elapsed_mins'] = out_df['time'].astype('timedelta64[m]')
        return out_df

    def _get_gene_expression_long_form(self):
        assert self.has_been_clustered, ('Cluster genes first, '
                                         'you hovercraft full of eels. '
                                         'I.e. call the .do_hierarchical_clustering() method before calling this method.')
        # TODO create private method of this snippet of code is used twice atm
        # Drop final column and add it later
        df_no_cluster_column = self.df.copy()
        df_no_cluster_column = df_no_cluster_column[df_no_cluster_column.columns[:-1]]

        # First extract meaningful information from column names,
        # so we can group by time and merge biological samples
        column_info = get_info_from_gse5628(df_no_cluster_column)
        column_tuples = list(zip(df_no_cluster_column, *column_info.values()))
        # Ensure we do not accidentally modify the original dataframe
        df_no_cluster_column.columns = pd.MultiIndex.from_tuples(
            column_tuples, names=['sample_name', 'time', 'tissue', 'replicate'])
        stacked_df = df_no_cluster_column.stack(level=['time', 'replicate'])
        # TODO check if only 1 non-NaN item per row
        # Sorry about how ugly this is, it's Friday afternoon 😶
        expressions_per_gene = stacked_df.sum(axis=1)
        expression_df = pd.DataFrame(expressions_per_gene, columns=['expression'])
        expression_df.reset_index(inplace=True)
        # TODO convert time to minutes
        # TODO currently this step is really slow, can definitely be sped up
        expression_df['cluster'] = expression_df.ID_REF.apply(lambda x: self.get_cluster_per_gene()[x])
        expression_df['time'] = expression_df['time'] / np.timedelta64(1, 'h')
        return expression_df


    def get_lpan_input_modules(self, n_clusters: int) -> pd.DataFrame:
        """For gene modules, get output that can be used to input into the Rscript LPAN workflow.
        (https://github.com/LiLabAtVT/LPANetwork)

        :param n_clusters: Number of clusters.
        :return: Output that resembles that can be used for lpan.
        """
        some_df = self.extract_module_expressions(n_clusters,
                                                  do_plotting=False)
        output_df = some_df.pivot(index='cluster_id', columns='sample',
                                  values='expression')
        output_df.index = self.module_prefix + output_df.index.astype(str)
        return output_df

    def get_lpan_input_tfs(self) -> pd.DataFrame:
        """Get expressions of transcription factors to be used in Rscript LPAN.
        Only use this if the object contains only transcription factors.
        I.e. the object has been created from a .split_off_tfs() method call.
        """
        output_df = self.df
        output_df.index = self.tf_prefix + output_df.index.astype(str)
        return output_df

    def split_off_tfs(self, path_to_tf_file: Path) \
            -> tuple[ExpressionMatrixTimeSeries, ExpressionMatrixTimeSeries]:
        """Split time series into set of transcription factors, and set of non-
        transcription factors

        :param path_to_tf_file: Path to file which contains list of
        transcription factor gene ids in column GeneID in tsv file.
        E.g. it can be downloaded from http://planttfdb.gao-lab.org/index.php?sp=Ath
        :return: Tuple of ExpressionMatrixTimeSeries non-tfs, and tfs.
        """
        tf_annotation_df = pd.read_csv(path_to_tf_file, sep='\t')
        is_tf = self.df.index.isin(tf_annotation_df.Gene_ID)
        tfs_df, non_tfs_df = self.df[is_tf], self.df[~is_tf]

        return (ExpressionMatrixTimeSeries(non_tfs_df),
                ExpressionMatrixTimeSeries(tfs_df))

    def merge_biological_samples(self) -> None:
        """Calculate average of two biological samples"""
        # First extract meaningful information from column names,
        # so we can group by time and merge biological samples
        column_info = get_info_from_gse5628(self.df.columns)
        column_tuples = list(zip(self.df.columns, *column_info.values()))
        # Ensure we do not accidentally modify the original dataframe
        temp_df = self.df.copy()
        temp_df.columns = pd.MultiIndex.from_tuples(
            column_tuples, names=['sample_name', 'time', 'tissue', 'replicate'])
        my_grouping = temp_df.groupby('time', axis=1)
        # Keep original sample names, even after grouping by time
        # (needed for compatibility) TODO with what?
        sample_names = [v[0][0] for (_, v)
                        in temp_df.groupby('time', axis=1).groups.items()]
        merged_samples = my_grouping.mean()
        merged_samples.columns = sample_names
        self.df = merged_samples

    def get_clusters_expressions_with_time(self, n_clusters: int) \
            -> tuple[np.ndarray, np.ndarray]:
        """For fitting ODEs, get expression of clusters over time.
        First array in tuple indicates time_points in minutes, second array
        indicates module expressions.
        """
        some_df = self._get_cluster_expression_long_form(n_clusters)
        new_df = some_df.pivot(index='cluster_id', columns='elapsed_mins',
                               values='expression')
        time_points = new_df.columns.to_numpy()
        module_expressions = new_df.to_numpy()
        return time_points, module_expressions
