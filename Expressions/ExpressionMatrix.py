"""
Contains classes that contain matrices of gene expression levels.
"""
from __future__ import annotations

import copy
import logging
import random
import tempfile
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Callable, Dict, List, Tuple
import re
import subprocess

import dill as pickle
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import GEOparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import zscore, bootstrap
import qnorm
from scipy.integrate._ivp.ivp import OdeResult
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score

from helpers import calculate_coefficient_of_variation, calculate_qcd, mean_bootstrap_error

class AggregationMethod(Enum):
    """Used to set the type of aggregation methot that is used
    to represent a module as one number. Can be either through
    the eigengene (value of PC1), or through the mean of a module.
    """
    EIGENGENE = 'eigengene'
    MEAN = 'mean'

class ExpressionMatrixTimeSeries:
    # Default prefixes to use when exporting to files
    tf_prefix = 'TF_'
    module_prefix = 'MODULE'

    def __init__(self, df: pd.DataFrame,
                 aggregation_method: AggregationMethod = AggregationMethod.MEAN):
        """Create expression matrix directly from Pandas dataframe

        :param df: Dataframe which contains gene expressions for various
                   biological samples
        """
        self.df = df
        self.df.index = self.df.index.str.upper()
        # Drop duplicates
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.has_been_clustered = False
        self.column_parser: Callable = None
        # TODO declare these from the start?
        # Can be mean or eigengene
        self.scale_before_analyses = True
        self.has_been_scaled = False
        self.aggregation_method = aggregation_method
        self.condition_names = None

    def __repr__(self):
        return (f'ExpressionMatrix with {len(self.df)} genes'
                f' and columns {self.df.columns.to_list()}')

    @classmethod
    def from_xlsx(cls, file_path, gpl_path: str = None,
                  log2_transform: bool = False):
        df = pd.read_excel(file_path, index_col=0)
        return cls._from_df_to_object(df, gpl_path, log2_transform)

    @classmethod
    def from_csv(cls, file_path: Path | str, log2_transform: bool = False,
                 sep: str = ',', gpl_path: str = None):
        """Create ExpressionMatrix from csv with genes in row and expression per sample in col

        :param file_path: path to csv
        :param log2_transform: if true, do log2 transform
        :param sep: csv separator
        :param gpl_path: Path of gene expression omnibus platform.
        If provided, use this to convert probe names to gene IDs.
        Download from e.g. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL198
        """
        df = pd.read_csv(file_path, sep=sep, index_col=0)
        return cls._from_df_to_object(df, gpl_path, log2_transform)

    @classmethod
    def _from_df_to_object(cls, df: pd.DataFrame, gpl_path: str, log2_transform: bool):
        """From a dataframe, construct ExpressionMatrix object"""
        if log2_transform:
            df = np.log2(df)
        if gpl_path:
            gpl = GEOparse.get_GEO(filepath=gpl_path, silent=True)
            translation_table = gpl.table
            df = cls._do_gpl_annotation(translation_table, df)
        return cls(df)

    @classmethod
    def from_geo_file(cls, file_path: str, annotate_from_gpl: bool = False,
                      log2_transform: bool = False, name_to_drop: str = None):
        """From a file path, correctly parse GEO expression file and
        return ExpressionMatrix object.

        :param annotate_from_gpl: If true, use the provided GPL annotation in
        based on the soft file to convert from probe ID to gene name.
        :param file_path: path to GEO expression file. Works on .soft format,
                      others have not been tested.
        :param array_annotation: Object which maps probe names of AGI names.
                                 Should have probe_to_agi() method.
        :param log2_transform: If true, Log2 transform the expression data.
        :param name_to_drop: Optional. If provided, drops all probes
                              that contain this name_to_drop in their name.
                              E.g. for affymetrix microarrays, one might
                              provide AFFX as a name to drop,
                              because all control probes contain this name.
        """
        gse = GEOparse.get_GEO(filepath=file_path, silent=True)
        # Merge all samples into one dataframe
        df = gse.pivot_samples("VALUE")
        # Probe ID must be uppercase
        df.index = df.index.str.upper()
        if name_to_drop:
            logging.info(f'Dropping all probes that contain {name_to_drop}')
            df = df.loc[df.index.map(lambda x: name_to_drop not in x), :]

        if annotate_from_gpl:
            assert len(gse.gpls) == 1, "GSE contains more than one platform?"
            # Take first (and only) value from dict
            gpl_object = sorted(gse.gpls.values())[0]
            gpl_table = gpl_object.table
            df = cls._do_gpl_annotation(gpl_table, df)

        if log2_transform:
            df = np.log2(df)
        # Convert sample names to titles that humans can understand
        better_name_dict = gse.phenotype_data.title.to_dict()
        df.columns = [better_name_dict[old_col] for old_col in df.columns]
        return cls(df)

    @classmethod
    def _do_gpl_annotation(cls, gpl_table, df):

        # Probe IDs must be uppercase
        gpl_table['ID'] = gpl_table['ID'].str.upper()
        df.index = df.index.str.upper()
        if 'AGI' in gpl_table.columns:
            true_name_col_name = 'AGI'
        else:
            true_name_col_name = "ORF"
        df = df.merge(gpl_table[['ID', true_name_col_name]], left_index=True, right_on='ID')
        # Drop NA mappings
        logging.info(f'{len(df)} Probe IDs at start')
        df = df.dropna(subset=true_name_col_name)
        logging.info(f'{len(df)} genes mapped to probe IDs')
        df = df.set_index(true_name_col_name)
        # Remove old probe ID column
        df = df.drop('ID', axis=1)
        return df

    def plot_corr_distribution(self, out_path: Path = None):
        """Plot the distribution of correlations between all genes,
        can be used to select a proper cutoff
        """
        correlation_matrix = np.corrcoef(self.df)
        upper_tri = np.triu(correlation_matrix, k=1)
        flat_values = upper_tri[np.nonzero(upper_tri)]
        sns.set_style('ticks')
        sns.histplot(flat_values, element='step')
        plt.xlabel('Pairwise pearson correlation')
        sns.despine()
        if out_path:
            plt.savefig(out_path)
        plt.show()

    def save_edgelist_for_cytoscape(self,
                                    out_path: Path,
                                    correlation_cutoff: float,
                                    abs_correlation: bool = True):
        """Save edgelist file that can be visualised with cytoscape

        :param out_path: path to save output file (.tsv file format)
        :param correlation_cutoff at this cutoff, genes are assigned an edge
        :param abs_correlation: If true, compare absolute correlation to
        cutoff instead of value between -1 and 1
        """
        correlation_matrix = np.corrcoef(self.df)
        # Find the correlations to check
        corr_to_check = np.triu(correlation_matrix, k=1)
        if abs_correlation:
            corr_to_check = abs(corr_to_check)
        # Select pairs above cutoff
        mask = corr_to_check > correlation_cutoff
        indices = np.where(mask)
        # Convert to edgelist
        gene_names = self.df.index
        # In case there are multiple gene names, just take the first one
        clean_gene_names = [name.split(';')[0] if ';' in name else name
                            for name in gene_names]
        pairs = [(clean_gene_names[i], clean_gene_names[j], correlation_matrix[i, j]) for
                 i, j in zip(*indices)]
        # Save gene pairs to a file
        with out_path.open('w+') as f:
            f.write("Gene1\tGene2\tCorr_strength\n")
            for pair in pairs:
                f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\n")

    def get_sample_names(self) -> np.array:
        """Returns all names of samples"""
        return self.df.columns.to_numpy()

    def get_gene_names(self) -> list:
        """Returns names of all gene names"""
        return self.df.index.to_list()

    def keep_only_samples_with_string(self, string_to_select: str) -> None:
        """Keep only columns that contain a certain substring
        """
        col_mask = [col for col in self.df.columns
                    if (string_to_select in col or 'cluster_id' in col or 'zero' in col)]
        self.df = self.df[col_mask]

    def plot_per_gene_std(self) -> pd.Series:
        """Plot per gene standard deviation across samples"""
        std_series = self.df.std(axis=1)
        sns.histplot(std_series)
        plt.show()
        return std_series

    def plot_sample_gene_heatmap(self, standard_scale=0) -> None:
        if self.has_been_clustered:
            df = self.df.drop('cluster_id', axis=1)
        sns.clustermap(df, method='complete', metric='correlation',
                       yticklabels=False, xticklabels=True,
                       standard_scale=standard_scale)
        plt.show()

    def get_module_sizes(self):
        assert self.has_been_clustered
        return self.df.groupby('cluster_id').apply(len)

    def plot_cluster_sizes(self, out_path: Path|None = None):
        sns.histplot(self.df.cluster_id.value_counts())
        plt.xlabel('Cluster size')
        if not out_path:
            plt.show()
        else:
            plt.savefig(out_path)
        plt.close()

    def save_for_limma(self, out_path: Path):
        self.df.to_csv(out_path)

    def assign_clusters_from_tf2_input(self, tf2_input_path: Path,
                                       overwrite: bool):
        """Find which genes belong to which cluster, based
        on the TF2Network input file.

        :param overwrite: If true, overwrite the existing clusters
        :param tf2_input_path: Path to tf2network input file, should
                                contain lines with cluster_id and gene
                                name separated by space.
        """
        mapping_df = pd.read_csv(tf2_input_path, sep=' ', names=['cluster_id','gene_name'], index_col='gene_name')
        self._apply_cluster_mapping_from_df(mapping_df, overwrite=overwrite)


    def _apply_cluster_mapping_from_df(self, mapping_df: pd.DataFrame,
                                       overwrite: bool = False):
        """

        :param overwrite: If true, overwrite existing clusters. If false and
                          df has already been clustered, method will raise
                          ValueError
        :param mapping_df: Index should be gene name, column should be 'cluster_id'
        """
        assert not mapping_df.index.has_duplicates, 'Mapping dataframe contains duplicate indices'
        # TODO is this proper way to handle mismatch?
        self.df.index = [i.split(';')[0] for i in self.df.index]
        # Drop duplicates based on index
        self.df = self.df.reset_index().drop_duplicates(
            subset='index').set_index('index')
        if 'cluster_id' in self.df.columns:
            if not overwrite:
                raise ValueError('DataFrame has already been clustered, '
                                 'but overwrite=False for the cluster '
                                 'assignment. To fix, assign '
                                 'the clusters with overwrite=True')
            else:
                self.df = self.df.drop('cluster_id', axis=1)
        new_df = pd.concat([self.df, mapping_df], join='inner', axis=1)
        logging.info(f'Lost {len(mapping_df)-len(new_df)} genes during annotation')
        self.df = new_df
        self.has_been_clustered = True

    def extract_module_expressions_long_form(self) -> pd.DataFrame:
        # Make sure clustering has been performed
        if not self.has_been_clustered:
            logging.warning('Not clustered yet=')
        # expressions = self.
        # match self.aggregation_method:
        #     case AggregationMethod.MEAN:
        #         func = self._get_mean_over_time
        #     case AggregationMethod.EIGENGENE:
        #         func = self._get_eigengene_over_time
        #     case _:
        #         raise NotImplementedError(
        #             f'{self.aggregation_method=} is not implemented')
        expressions = self.df.groupby('cluster_id').apply(
            self._aggregate_module_expressions_one_group)
        # Convert to long form
        expressions = expressions.T.reset_index().melt(
            id_vars=['time', 'condition'], value_name='expression')

        # Get time point info from sample names
        # Parse time info based on columns
        if 'time' not in expressions.columns:
            new_cols = self.column_parser(expressions['sample'].to_list())
            expressions = pd.concat(
                [expressions, pd.DataFrame.from_dict(new_cols)], axis=1)
        expressions['elapsed_mins'] = expressions['time'].dt.total_seconds() / 60
        return expressions

    def _get_eigengene_over_time(self,
                                 one_group_df: pd.DataFrame) -> pd.Series:
        """Get the series of the module eigengene, to see how it behaves over
           time. Only applies this to one module, so recommended calling this
           from a grouped_df.apply context.

        :param one_group_df: Dataframe of one module to apply eigengene
                             methodology to.
        :return: expression of module eigengene at all time points
        """
        one_group_df = one_group_df.drop('cluster_id', axis=1)
        # Calculate PCA
        pca = self._do_pca_of_group(one_group_df)
        # one_group_transpose = one_group_df.T
        # if self.scale_before_analyses:
        #     one_group_transpose = one_group_transpose - one_group_transpose.mean()
        #     one_group_transpose = one_group_transpose / one_group_transpose.std()

        eigen_values_through_time = pca.transform(one_group_df.T).flatten()
        # Set correlation between eigengene and mean expression to be positive
        # since direction of PC is arbitrary
        eigen_value_to_mean_expression_corr = np.corrcoef([one_group_df.mean(),
                                                           eigen_values_through_time])
        if eigen_value_to_mean_expression_corr[0, 1] < 0:
            eigen_values_through_time = eigen_values_through_time * -1
        eigen_values_through_time = pd.Series(eigen_values_through_time)
        updated_indices = pd.DataFrame(
            self.column_parser(one_group_df.columns))
        if 'rep_nr' in updated_indices:
            updated_indices = updated_indices.drop('rep_nr', axis=1)
        eigen_values_through_time.index = pd.MultiIndex.from_frame(updated_indices)
        return eigen_values_through_time

    def _get_mean_over_time(self, one_group_df: pd.DataFrame) -> pd.Series:
        """Get the series of the module mean, to see how it behaves over
           time. Only applies this to one module, so recommended calling this
           from a grouped_df.apply context.

        :param one_group_df: Dataframe of one module to apply mean
                             methodology to.
        :return: expression of module mean at all time points
        """
        if not self.has_been_scaled:
            logging.warning("Extracting mean of module even though it hasn't been scaled before")
        one_group_df = one_group_df.drop('cluster_id', axis=1)
        # one_group_df = one_group_df.T
        # if self.scale_before_analyses:
        #     one_group_df = one_group_df - one_group_df.mean()
        #     one_group_df = one_group_df / one_group_df.std()

        expressions = one_group_df.mean()
        expressions = self._annotate_sample_names_expressions(expressions)
        return expressions

    def _annotate_sample_names_expressions(self, expressions):
        updated_indices = pd.DataFrame(
            self.column_parser(expressions.index))
        if 'rep_nr' in updated_indices:
            updated_indices = updated_indices.drop('rep_nr', axis=1)
        expressions.index = pd.MultiIndex.from_frame(
            updated_indices)
        return expressions

    def _do_pca_of_group(self, one_group_df: pd.DataFrame) -> PCA:
        """For one gene module, do PCA with one component to get idea of
        eigengene values and how much variance it explains

        :param one_group_df: Dataframe that should come from one cluster.
        :return: PCA object that was fitted to input dataframe
        """
        assert self.has_been_scaled
        pca = PCA(n_components=1)
        transposed_df = one_group_df.T
        # if self.scale_before_analyses:
        #     transposed_df = transposed_df - transposed_df.mean()
        #     transposed_df = transposed_df / transposed_df.std()
        pca.fit(transposed_df)
        return pca

    def save_tf_produced_by_module_file(self, out_file_path: Path,
                                        tf_list_path: Path):
        """Save edge list that maps transcription factors to their module

        :param out_file_path: Path to output csv file.
        :param tf_list_path: Path to list of transcription factors. Must contain
        a column called 'Gene_ID'.
        """
        if not tf_list_path.exists():
            # Search one directory higher (I know this is ugly btw)
            tf_list_path = '..' / tf_list_path
        # Get transcription factors
        tf_df = pd.read_csv(tf_list_path, sep='\t')
        tf_list = tf_df['Gene_ID'].to_list()
        # Find which transcription factors are in the expressionmatrix
        out_df = self.df[self.df.index.str.upper().isin(tf_list)].copy()
        assert len(out_df) > 1, 'Got an empty selection of transcription factors'
        # Add prefixes to modules and transcription factors
        out_df.index = self.tf_prefix + out_df.index.astype(str)
        out_df['cluster_id'] = self.module_prefix + out_df.cluster_id.astype(str)
        out_file_path.parent.mkdir(exist_ok=True, parents=True)
        out_df.to_csv(out_file_path, sep=' ', columns=['cluster_id'],
                      header=False)

    def get_cluster_per_gene(self) -> dict[str, int]:
        """For each gene, get its cluster_id. Can only be called after
        self.do_hierachical_clustering() has been called.

        :returns: Dict with gene name as key and cluster_ID as value.
        """
        assert 'cluster_id' in self.df.columns,\
                'Run do_hierachical_clustering() first!'
        return self.df.cluster_id.to_dict()

    def get_genes_per_cluster(self) -> dict[int, list[str]]:
        """For each cluster, get Gene IDs. Can only be called after
        self.do_hierachical_clustering() has been called.

        :returns: Dict with keys cluster_id and values a list of all genes in that cluster .
        """
        assert 'cluster_id' in self.df.columns,\
                'Run do_hierachical_clustering() first!'
        # Ensure that the genes are returned as list, and not an index object
        return {cluster_id: genes.tolist()
                for cluster_id, genes
                in self.df.groupby('cluster_id').groups.items()}

    def save_distance_matrix(self, out_path: Path):
        """"""
        dist = self.get_distance_matrix()
        dist.to_pickle(out_path)

    def get_distance_matrix(self, absolute_dist=False) -> pd.DataFrame:
        """Get correlation-based pairwise distance"""
        if not absolute_dist:
            dist = pdist(self.df, metric='correlation')
            square_dist = squareform(dist)
        else:
            square_dist = self.df.T.corr()
            square_dist = 1 - abs(square_dist)

        dist_df = pd.DataFrame(square_dist, index=self.df.index, columns=self.df.index)
        return dist_df

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
        # # Calculate pearson correlation
        # subset_corr = np.corrcoef(self.df)
        # # Calculate distance
        # dist = 1 - subset_corr
        #
        # # subset_corr = np.abs(subset_corr)
        # # subset_corr = subset_corr**2
        # # Squareform diagonality checking can be too strict, so we do it
        # # explicitly here and disable checks in the squareform function call
        # assert np.allclose(dist, dist.T), 'Matrix does not appear symmetrical?'
        # assert sum(np.diag(dist)) < 1e-6, 'Sum of diagonal too high'
        # dist = squareform(dist, checks=False)

        dist = pdist(self.df, metric='correlation')
        # Create linkage matrix and infer clusters
        linkage_matrix = linkage(dist, method='complete')
        # linkage_matrix = linkage(dense_dist, method='average')
        clustering = fcluster(linkage_matrix, n_cluster, 'maxclust')
        self.df = self.df.assign(cluster_id=clustering)
        self.has_been_clustered = True

    @classmethod
    def from_simulated_data(cls, sim_data: OdeResult):
        """Create df where each gene is just representative of one module,
        so you can feed simulated data into all the algorithms that
        need ExpressionMatrixTimeSeries as an input.
        """
        df = pd.DataFrame.from_records(sim_data.y)
        # Dummy column names
        df.columns = [f'AtGen_6-9711_Simulated-Shoots-{t}h_Rep1'
                      for t in sim_data.t]
        created_object = cls(df)
        created_object.df['cluster_id'] = df.index.to_list()
        created_object.has_been_clustered = True

        return created_object

    def plot_clusters_over_time(self, plot_units: bool = False,
                                out_path: Path | None = None) -> None:

        """Plot expression of clusters over time.

        :param split_by_condition: If multiple conditions in dataframe,
            split into different plots using these keywords
        :param plot_units: If true, plot line for each gene individually.
                            If false, plot mean of all genes in a cluster.
        :param timescale: If 'days', plot time in days, if 'hours' plot time in hours
        """
        sns.set_theme()
        if plot_units:
            # sns.relplot(data=some_df, x='time (days)', y='expression',
            #             kind='line',
            #             hue='replicate', col='cluster_id',
            #             palette=sns.color_palette(),
            #             units='level_0', estimator=None, lw=1, alpha=.2)
            some_df = self._get_gene_expression_long_form()
            some_df['time (days)'] = some_df['time'].dt.days
            nr_hues = some_df['replicate'].nunique()
            sns.relplot(data=some_df, x='time (days)', y='expression',
                        kind='line',
                        hue='replicate', col='cluster_id',
                        palette=sns.color_palette(n_colors=nr_hues),
                        units='ID_REF', estimator=None, lw=1, alpha=.2)
        else:
            expressions = self.extract_module_expressions_long_form()
            expression_list = self.split_series_into_different_conditions(
                expressions)
            expressions = pd.concat(expression_list)
            sns.relplot(data=expressions, x='time',
                         y='expression', col='cluster_id', hue='condition',
                        col_wrap=4, kind='line')
            # plt.title(condition_name)
            if out_path:
                raise NotImplementedError("File saving not implemented yet")
            else:
                plt.show()
            plt.close()

    def plot_per_gene_mad(self) -> pd.Series:
        """Plot per gene mean absolute deviation across samples"""
        mad_series = self._calculate_per_gene_statistic('mad')
        sns.histplot(mad_series)
        plt.show()
        return mad_series

    def scatterplot_of_two_per_gene_stats(
            self,
            stat1: Literal['std', 'mean', 'mad', 'cv', 'qcd', 'cond_rmse'],
            stat2: Literal['std', 'mean', 'mad', 'cv', 'qcd', 'cond_rmse'],
            title: str = None,
            out_path: str = None,
            plotting_func = sns.scatterplot
    ) -> pd.DataFrame:
        """Show a scatterplot between summary statistics of each gene.
        E.g. between mean expression and its standard deviation

        :param stat1: Statistic shown on x-axis
        :param stat2: Statistic shown on y-axis
        """
        stat_name_list = [stat1, stat2]
        out_list = []
        for stat_name in stat_name_list:
            stat_list = self._calculate_per_gene_statistic(stat_name)
            stat_list.name = stat_name
            out_list.append(stat_list)

        merged_df = pd.concat(out_list, axis=1, join='inner')
        plotting_func(data=merged_df, x=stat1, y=stat2)
        if title:
            plt.suptitle(title)
        if out_path:
            plt.savefig(out_path)
            plt.close()
        else:
            plt.show()
        return merged_df

    def _calculate_per_gene_statistic(
            self,
            method: Literal['std', 'mad', 'cv', 'qcd', 'cond_rmsd',
            'norm_cond_rmsd', 'mean']
    ) -> pd.Series:
        """Calculate for each gene a statistic over all samples

        :param method: Method to calculate over samples:
                        std: standard deviation
                        mad: mean absolute deviation
                        cv: coefficient of variation
                        qcd: quartile coefficient of dispersion
                        cond_rmse: root mean square error of expression
                        norm_cond_rmse: root mean square error of expression
                            divided by standard deviation over all samples
                        between treatment and control
                        mean: mean expression over all samples
        :return: Series containing output result for each gene
        """

        if method in ['qcd', 'cv']:
            # Cannot take negative values:
            # Subtract minimum value to all values, so qcd / cv does not
            # get negative inputs
            min_value = self.df.min().min()
            if min_value < 0:
                # all_num_df = self.df.apply(pd.to_numeric)
                translated_df = self.df - min_value
            else:
                translated_df = self.df

        # # Random snippet to plot compare measures
        # self.df['qcd'] = translated_df.apply(calculate_qcd, axis=1)
        # self.df['cv'] = translated_df.apply(calculate_coefficient_of_variation,
        #                               axis=1)
        # fig, axs = plt.subplots(2, 1)
        # sns.stripplot(data=self.df, x='qcd', ax=axs[0], alpha=.5)
        # sns.stripplot(data=self.df, x='cv', ax=axs[1], alpha=.5)
        # plt.tight_layout()
        # plt.show()
        # logging.info(self.df[['qcd', 'cv']].corr('spearman'))

        match method:
            case 'mean':
                return self.df.mean(axis=1)
            case 'cond_rmsd':
                return self.get_gene_rmse_difference_between_conditions()
            case 'std':
                return self.df.std(axis=1)
            case 'mad':
                return self.df.mad(axis=1)
            case 'qcd':
                return translated_df.apply(calculate_qcd, axis=1)
            case 'cv':
                return translated_df.apply(calculate_coefficient_of_variation,
                                           axis=1)
            case _:
                raise NotImplementedError(f'{method} not available')

    def _get_gene_expression_long_form(self):
        """Expression of all individual genes.
        Used if you want to plot them all seperately
        """
        assert self.has_been_clustered, 'Cluster genes first.'
        # Drop final column and add it later
        df_no_cluster_column = self.df.copy()
        df_no_cluster_column = df_no_cluster_column[df_no_cluster_column.columns[:-1]]

        # First extract meaningful information from column names,
        # so we can group by time and merge biological samples

        column_info = self.column_parser(df_no_cluster_column)
        column_tuples = list(zip(df_no_cluster_column, *column_info.values()))
        # Ensure we do not accidentally modify the original dataframe
        #
        df_no_cluster_column.columns = pd.MultiIndex.from_tuples(
            column_tuples, names=['sample_name', 'time', 'condition', 'replicate'])
        stacked_df = df_no_cluster_column.stack(level=['time', 'replicate'])
        # TODO check if only 1 non-NaN item per row
        # Sorry about how ugly this is, it's Friday afternoon 😶
        expressions_per_gene = stacked_df.sum(axis=1)
        expression_df = pd.DataFrame(expressions_per_gene, columns=['expression'])
        expression_df.reset_index(inplace=True)
        if expression_df.columns[0] != 'ID_REF':
            # Change column name to gene identifier again
            expression_df = expression_df.rename(
                columns={expression_df.columns[0]: 'ID_REF'}
            )
        expression_df = expression_df.merge(self.df['cluster_id'],
                                            left_on='ID_REF', right_index=True)
        return expression_df

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
        assert is_tf.any(), 'No transcription factors in the modules. ' \
                            'Consider increasing the number of genes that you consider'
        tfs_df, non_tfs_df = self.df[is_tf], self.df[~is_tf]

        return (ExpressionMatrixTimeSeries(non_tfs_df),
                ExpressionMatrixTimeSeries(tfs_df))

    def merge_biological_samples(self) -> None:
        """Calculate average of all biological samples"""
        # First extract meaningful information from column names,
        # so we can group by time and merge biological samples
        if self.has_been_clustered:
            clustering_list = self.df['cluster_id']
            self.df = self.df.drop('cluster_id', axis=1)
        assert self.column_parser is not None, "No column parsing function has been specified"
        column_info = self.column_parser(self.df.columns)
        column_tuples = list(zip(self.df.columns, *column_info.values()))
        # Ensure we do not accidentally modify the original dataframe
        temp_df = self.df.copy()
        temp_df.columns = pd.MultiIndex.from_tuples(
            column_tuples, names=['sample_name', 'time', 'condition', 'replicate'])
        my_grouping = temp_df.T.groupby(['time', 'condition'])
        # Keep original sample names, even after grouping by time
        sample_names = [v[0][0] for (_, v)
                        in my_grouping.groups.items()]
        merged_samples = my_grouping.mean().T
        merged_samples.columns = sample_names
        if self.has_been_clustered:
            self.df = pd.concat([merged_samples, clustering_list], axis=1)
        else:
            self.df = merged_samples

    def get_clusters_expressions_with_time(
            self) -> pd.DataFrame:
        """For fitting ODEs, get expression of clusters over time.

        Only run this if expression module only contains samples of
        one condition (because of self.keep_only_samples_with_string())

        :return: DataFrame with columns as timepoints in hours and rows as gene modules
        """
        module_expressions = self.extract_module_expressions_long_form()
        module_expressions = module_expressions.pivot(
            index='cluster_id', columns='elapsed_mins', values='expression')
        time_points = module_expressions.columns.to_numpy()
        # Convert time to hours
        time_points = time_points / 60
        # module_expressions = module_expressions.to_numpy()
        module_expressions.columns = time_points
        return module_expressions

    def write_tf2_input_file(self, out_path: Path,
                             omit_unannotated_genes: bool = True,
                             cut_gene_names: bool = True):
        """Create file that can be pasted into the TF2network website
        (http://bioinformatics.psb.ugent.be/webtools/TF2Network/index.php).

        :param out_path: Filename of output txt file
        :param omit_unannotated_genes: If true, do not save genes that could
        not be annotated (names like '246771_at') because TF2NETWORK won't know
        what these genes are.
        """
        genes_with_clusters = self.get_cluster_per_gene()
        lines = []
        for gene_name, cluster_id in genes_with_clusters.items():
            if cut_gene_names and ';' in gene_name:
                # In case there are multiple gene names, just take the first one
                gene_name = gene_name.split(';')[0]
            if omit_unannotated_genes and not gene_name.upper().startswith('AT'):
                # Gene could not be annotated, so tf2 won't find it
                continue
            lines.append(f'{cluster_id} {gene_name}\n')
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.open('w+') as f:
            f.writelines(lines)

    def get_correlation(self, module_index: int, tf_name: str,
                        plot: bool = False, method: str = 'pearson') -> float:
        """Calculate the correlation between eigengene of a module and a
        transcription factor

        :param module_index: the index of the module (e.g. 1)
        :param tf_name: Name of transcription factor
        :param method: Method to calculate correlation.
                       E.g. spearman or pearson.
        :param plot: If true, plot the distribution of correlations
        :return: Pearson correlation coefficient
        """
        # TODO eventually check if this can be moved higher up in the hierarchy
        assert self.has_been_clustered, 'Cluster object first'
        if tf_name not in self.df.index:
            logging.warning(f'TF ({tf_name}) not present in dataframe')
            return np.nan
        module_expressions = self.df[self.df['cluster_id'] == module_index]
        module_expressions = self._aggregate_module_expressions_one_group(
            module_expressions)
        tf_expressions = self.df.loc[tf_name]
        tf_expressions = tf_expressions.drop('cluster_id')
        # Ensure that indices match between eigengene and transciption factor
        # Otherwise cannot calculate correlation
        updated_indices = pd.DataFrame(
            self.column_parser(tf_expressions.index))
        if 'rep_nr' in updated_indices:
            updated_indices = updated_indices.drop('rep_nr', axis=1)
        tf_expressions.index = pd.MultiIndex.from_frame(updated_indices)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(module_expressions.to_numpy(),
                    tf_expressions.to_numpy(), 'o')
            ax.set_xlabel(f'Module {module_index} expression')
            ax.set_ylabel(f'{tf_name} expression')
            plt.show()
        corr =  tf_expressions.corr(module_expressions, method=method)
        return corr

    def _get_characteristics_of_clusters(self, tf2_output: Path) -> pd.DataFrame:
        """Retrieve how the clusters behave in the dataset

        Returns a df containing
            - the percentage of explained variance of the first PC to measure
              expression coherence
            - Mean pairwise absolute correlation to also measure
              expression coherence
            - Size of the cluster
            - How much its expression changes over time
            - Difference in expression between control condition and
              experimental condition
            - Median expression of genes in module to see if it is switched on
            - Correlation to a phenotype (in self.phenotype_dict)
        """
        grouped_df_unnormalised = self.df.groupby('cluster_id')
        mean_expression = grouped_df_unnormalised.mean().T.mean()
        if self.scale_before_analyses:
            self.do_z_scaling()
        grouped_df = self.df.groupby('cluster_id')
        characteristics_dict = {
            'explained_var': self._get_eigengene_explained_var,
            'size': len,
            #'corr_to_phenotype': self._corr_to_phenotypes,
            'mean_pairwise_abs_cor': self._mean_pairwise_abs_cor,
            # 'var_through_time': self._get_module_variation_over_time,
            'difference_between_conditions': self._get_module_difference_between_conditions,
        }
        all_dfs = []
        for col_name, function_name in characteristics_dict.items():
            charac_series = grouped_df.apply(function_name)
            charac_series.name = col_name
            all_dfs.append(charac_series)
        summary_df = pd.concat(all_dfs, axis=1)


        summary_df = summary_df.assign(mean_expression=mean_expression)

        if tf2_output.exists():
            tf2_df = pd.read_csv(tf2_output, sep='\t')
            # This used to be binary, now changing it to total nr of tfbs
            # Old yes/no implementation here
            # tfbs_score = summary_df.index.isin(tf2_df['GeneSet']).astype(int)
            # New implementation here
            tfbs_score = []
            nr_tfbs_per_module_dict = tf2_df['GeneSet'].value_counts().to_dict()
            for module_idx in summary_df.index:
                nr_tfbs = nr_tfbs_per_module_dict.get(module_idx, 0)
                tfbs_score.append(nr_tfbs)
        else:
            logging.info(f'No tf2output found at {tf2_output}, z-scoring will assume that '
                         'none of the modules have a tfbs. '
                         'If you have tf2output, rename it to be this file.')
            tfbs_score = 0
        summary_df = summary_df.assign(tfbs_present=tfbs_score)
        return summary_df

    def _get_module_difference_between_conditions(self, one_group_df: pd.DataFrame):
        """Measure difference in expression of the module between two conditions

        :param one_group_df: Dataframe that contains expressions of genes in
                             one column. (Typically from .grouppby() method)
        :return: mean squared error
        """
        expressions = self._aggregate_module_expressions_one_group(
            one_group_df)
        expressions.name = 'expressions'
        mse = self._calculate_mse_two_expression_series(expressions)
        return mse

    def get_gene_rmse_difference_between_conditions(self) -> pd.Series:
        """For each gene calculate the MSE of its expression in
        control vs treatment
        """
        annotated_df = self._annotate_sample_names_expressions(self.df.T)

        control_series = annotated_df[annotated_df.index.get_level_values(
            'condition').isin(['zero', self.condition_names[0]])]
        treatment_series = annotated_df[annotated_df.index.get_level_values(
            'condition').isin(['zero', self.condition_names[1]])]

        control_series.index = control_series.index.droplevel('condition')
        treatment_series.index = treatment_series.index.droplevel('condition')

        rmse = mean_squared_error(control_series, treatment_series,
                                  multioutput='raw_values',
                                    squared=False)
        rmse = pd.Series(rmse, index=annotated_df.columns)
        rmse.name = 'mean_square_error'
        return rmse

        # def small_changing_function(x):
        #     x.name = 'expressions'
        #     return self._calculate_mse_two_expression_series(x)
        # mse =  annotated_df.apply(small_changing_function)
        # # TO speed up first split the df into two, then do subtraction and squaring simultaneously
        # logging.info('Calculating difference between gene expression in '
        #              'samples, this is implemented quite poorly '
        #              'so will probably take a while 🙃')
        #
        # return mse

    def _calculate_mse_two_expression_series(self, expressions):
        # Split into drought and control time series
        assert len(self.condition_names) == 2, (
            f'Can only calculate difference '
            f'between two conditions.'
            f' Now provides with'
            f' {len(self.condition_names)}'
            f'conditions: {self.condition_names}.'
            f'Specify self.condition_names to get this workin properly.')
        control_series, drought_series = self.split_series_into_different_conditions(
            expressions)
        # Ensure that in similar order and everything matches
        merged = control_series.merge(drought_series, on='time',
                                      suffixes=['_control', '_condition'])
        mse = mean_squared_error(merged['expressions_control'],
                                 merged['expressions_condition'])
        return mse

    def split_series_into_different_conditions(self, expressions):
        out_list = []
        for condition_name in self.condition_names:
            if expressions.index.nlevels > 1:
                # Multilevel index
                series_of_condition = expressions[expressions.index.get_level_values(
                    'condition').isin(['zero', condition_name])].reset_index()
            else:
                series_of_condition = expressions[expressions['condition'].isin(['zero', condition_name])]
                series_of_condition.loc[:, 'condition'] = condition_name
            out_list.append(series_of_condition)
        return out_list

    def _aggregate_module_expressions_one_group(self, one_group_df):
        match self.aggregation_method:
            case AggregationMethod.EIGENGENE:
                expressions = self._get_eigengene_over_time(one_group_df)
            case AggregationMethod.MEAN:
                expressions = self._get_mean_over_time(one_group_df)
            case _:
                raise NotImplementedError
        return expressions

    @staticmethod
    def _mean_pairwise_abs_cor(one_group_df: pd.Dataframe) -> float | np.floating:
        """Get the mean pairwise absolute correlation between all variable.

        Method typically called in a grouped_df.apply()

        :param one_group_df: Dataframe that should come from one cluster.
        :return: mean correlation value
        """
        one_group_df = one_group_df.drop('cluster_id', axis=1)
        corr_matrix = np.corrcoef(one_group_df)
        corr_matrix = np.abs(corr_matrix)
        dense_corr = squareform(corr_matrix, checks=False)
        return np.mean(dense_corr)

    def _get_eigengene_explained_var(self, one_group_df: pd.DataFrame) -> float:
        """
        Get how much variance the first principal component explains.
        (Used to get an idea how much expression coherence
        is within the cluster)

        :param one_group_df: Dataframe that should come from one cluster.
        :return: explained variance as float
        """
        # Calculate PCA
        one_group_df = one_group_df.drop('cluster_id', axis=1)
        pca = self._do_pca_of_group(one_group_df)
        return pca.explained_variance_ratio_[0]

    def _get_module_variation_over_time(self,
                                        one_group_df: pd.DataFrame
                                        ) -> float | np.floating:
        """Standard deviation of module expression over time

        Used to see if a gene module changes expression throughout
        the different samples.

        :param one_group_df: Dataframe that should come from one cluster.
        :return: standard deviation of expression of module through time
        """

        expr_through_time = self._aggregate_module_expressions_one_group(
            one_group_df)
        return np.std(expr_through_time)

    def get_z_score_of_cluster_characteristics(
            self,
            tf2_output: Path | None,
            subset: Tuple[str] = (
                    'explained_var',
                    'difference_between_conditions',
                    # 'corr_to_phenotype',
                    'mean_expression',
                    'tfbs_present')
    ) -> pd.DataFrame:
        """For clusters, get their sum of z-scores to find out which is the
           most interesting to look at

        :param subset: Column names to use for z-score selection
        :param tf2_output: Path to TF2Network output file, used to see if
                           modules have an enriched TFBS
        :return: dataframe of z_scores for each module
        """
        summary_df = self._get_characteristics_of_clusters(tf2_output)
        summary_df = summary_df[list(subset)]
        # summary_df = summary_df.drop(['size',
        #                               'mean_pairwise_abs_cor'], axis=1)

        # Log transform number of tfbs to get more normal distribution
        summary_df['tfbs_present'] = np.log(summary_df['tfbs_present'] + 1)

        # Get Z scores
        z_scores = summary_df.apply(zscore)
        stouffler_z = z_scores.sum(axis=1) / np.sqrt(len(z_scores.columns))
        z_scores = z_scores.assign(z_sum=stouffler_z)
        return z_scores

    def keep_highest_z_clusters(self, nr_clusters: int,
                                tf2_output_path: Path | None,
                                plotting_path: Path | None = None
                                ) -> pd.DataFrame:
        """Only keep clusters with highest z_scores. Modifies object in-place.

        :param plotting_path: If True, show plots of z_scores
        :param nr_clusters: Top number of clusters to select
        :param tf2_output_path: Path to TF2Output, used to determine if module
                                contains TFBS (which has positive
                                impact on z-score).
        :return: Dataframe of z-scores of the best clusters
        """
        z_scores = self.get_z_score_of_cluster_characteristics(tf2_output_path)

        worst_clusters =  z_scores.sort_values(
            'z_sum', ascending=True).head(nr_clusters)
        best_clusters = z_scores.sort_values(
            'z_sum', ascending=False).head(nr_clusters)
        if plotting_path:
            # Is higher better for all? Or how can we make it that way?
            sns.pairplot(z_scores)
            plt.savefig(plotting_path / 'z_scores_pairplot.svg')
            plt.close()

            sns.histplot(z_scores, kde=True, element='step',
                         common_norm=False)
            plt.savefig(plotting_path / 'all_z_scores_hist.svg')
            plt.close()

            sns.histplot(z_scores['z_sum'])
            plt.savefig(plotting_path / 'summed_z_hist.svg')
            plt.close()
            # sns.boxplot(z_scores.sum(axis=1))
            # plt.show()

            sns.heatmap(z_scores.sort_values('z_sum', ascending=False))
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.savefig(plotting_path / 'heatmap_z_score_all_clusters.svg')
            plt.close()

            sns.heatmap(best_clusters, cmap='vlag', center=0)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.savefig(plotting_path / 'heatmap_z_score_best_clusters.svg')
            plt.close()

        self.df = self.df[self.df['cluster_id'].isin(best_clusters.index)]
        return best_clusters

    def get_pairwise_module_correlations(self) -> pd.DataFrame:
        """Get dataframe that shows pairwise module correlations"""
        match self.aggregation_method:
            case AggregationMethod.EIGENGENE:
                expressions = self.df.groupby('cluster_id').apply(
                    self._get_eigengene_over_time)
            case AggregationMethod.MEAN:
                expressions = self.df.groupby('cluster_id').mean()
            case _:
                raise NotImplementedError(f'{self.aggregation_method} not found as method')
        correlations = expressions.T.corr()
        return correlations

    def keep_only_modules_in_network(self, module_module):
        """Filters the expression matrix to keep only the modules present
        in the given module_module network.

        :param module_module: ModuleRegulatoryNetwork that only contains the modules you want to keep
        :type module_module: ModuleRegulatoryNetwork
        """
        self.df = self.df[
            self.df['cluster_id'].isin(
                [int(i.replace(module_module.module_prefix, ""))
                 for i in module_module.get_modules()])]

    def get_sem_per_cluster(self):
        assert self.has_been_clustered
        return self.df.groupby('cluster_id').sem()

    def get_std_per_cluster(self, mean_over_all_samples=False):
        """Standard deviation of each cluster

        :param mean_over_all_samples: If true, return the mean std over all samples
        If false, return a separate std for each module for each sample
        """
        assert self.has_been_clustered
        # Standard deviations for each module for each sample
        stdevs = self.df.groupby('cluster_id').std()
        if mean_over_all_samples:
            # Average the mean over all samples
            return stdevs.mean(axis=1)
        else:
            return stdevs

    def get_ci_per_cluster(self, confidence_level=.95):
        """Get confidence interval per cluster at each time point"""
        assert self.has_been_clustered
        logging.warning(f'Setting {confidence_level=} currently not passed to underlying method')
        return self.df.groupby('cluster_id').apply(mean_bootstrap_error)

    def get_all_explained_vars(self):
        assert self.has_been_clustered
        grouped_df = self.df.groupby('cluster_id')
        return grouped_df.apply(self._get_eigengene_explained_var)

    def post_to_tf2network(self, tf2_in_path: Path, tf2_out_path: Path):
        """Automatically post gene clustering to TF2Network"""
        assert tf2_in_path.parent == tf2_out_path.parent, "At the moment only having TF2 in and ouput in the same folder is supported"
        # Remove old file
        tf2_out_path.unlink(missing_ok=True)
        self.write_tf2_input_file(tf2_in_path)
        url = 'http://bioinformatics.psb.ugent.be/webtools/TF2Network/'
        chrome_options = Options()
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument('--no-sandbox')
        # chrome_options.add_argument('--remote-debugging-pipe')

        # ensure chrome and the chromedriver are installed and compatible
        # with each other

        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--accept-insecure-certs")
        chrome_options.add_argument("--ignore-certificate-errors")
        prefs = {
            "download.default_directory": str(tf2_in_path.parent.resolve())}
        chrome_options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(options=chrome_options)

        # # Disable https redirect
        # driver.get('chrome://net-internals/#hsts')
        # url_input = driver.find_element(By.ID, value='domain-security-policy-view-delete-input')
        # url_input.send_keys(url)
        # delete_button = driver.find_element(By.ID, value='domain-security-policy-view-delete-submit')
        # delete_button.click()
        #
        # driver.get('chrome://settings/clearBrowserData')
        # clear_button = driver.find_element(By.XPATH, value='//*[@id="clearBrowsingDataConfirm"]')
        # clear_button.click()

        driver.get(url)
        logging.warning('MANUALLY MOVE TO THE HTTP WEBSITE')
        with tf2_in_path.open('r') as f:
            input_text = f.read()
        file_input = driver.find_element(By.NAME, value='gene_set')
        file_input.send_keys(input_text)

        direct_check_box = driver.find_element(By.NAME, value='direct')
        direct_check_box.click()

        submit_button = driver.find_element(by=By.NAME, value="submit")
        submit_button.click()
        tick = datetime.now()
        interval_time = 10
        while not (tf2_out_path.parent / 'tf2network_output.tsv').exists():
            time.sleep(interval_time)
            total_time = datetime.now() - tick
            logging.info(f'Waited {total_time.total_seconds():.0f} s for download of TF2Network now')

        (tf2_out_path.parent / 'tf2network_output.tsv').rename(tf2_out_path.with_name(tf2_out_path.name))
        print()

    def assign_clusters_from_wgcna(self, wgcna_module_assignment):
        assignment_df = pd.read_csv(wgcna_module_assignment, index_col='gene_id', usecols=['gene_id','colors'])
        # Map colours to index
        mapping_dict = {colour: i for (i, colour) in enumerate(assignment_df['colors'].unique())}
        logging.debug(f"Colour mapping dict: {mapping_dict}")
        assignment_df['cluster_id'] = assignment_df['colors'].map(mapping_dict)
        assignment_df = assignment_df.drop('colors', axis=1)

        self._apply_cluster_mapping_from_df(assignment_df)

    def do_genewise_min_max_scaling(self):
        """Do min-max scaling for all genes"""
        # Correct axis
        if self.has_been_clustered:
            clusters = self.df['cluster_id']
            self.df = self.df.drop('cluster_id', axis=1)

        # self.df = self.df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        self.df = self.df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

        # assert np.all(self.df.mean(axis=1) < 0.01)

        if self.has_been_clustered:
            self.df['cluster_id'] = clusters
        self.has_been_scaled = True

    def get_correlation_matrix(self) -> pd.DataFrame:
        assert not self.has_been_clustered
        return self.df.T.corr()

    def get_similarity_matrix(self)-> pd.DataFrame:
        """Get signed similarity matrix that can be used to cluster in WGCNA
        """
        cor_matrix = self.get_correlation_matrix()
        similarity_matrix = (1 + cor_matrix) / 2
        return similarity_matrix

    def do_random_clustering_with_given_size_dist(self,
                                                  wgcna_label_file: Path | None,
                                                  use_own_clustering: bool = False):
        """Compute a random distribution with a given size distribution and
        see how that compares to the non-random clustering.

        Size distribution can be done based on an existing WGCNA label file (if provided)

        Alternatively, if a DF has already been clustered, it can be based on this own clustering

        :param wgcna_label_file: Path to WGCNA label file
        :param use_own_clustering: If True, do random clustering based on
        size distribution of how expressionmatrix is currently clustered.
        """
        if use_own_clustering and wgcna_label_file is None:
            assert self.has_been_clustered
            self.df['cluster_id'] = np.random.permutation(self.df['cluster_id'])
        elif use_own_clustering and wgcna_label_file is not None:
            raise ValueError(f'Cannot set {use_own_clustering=} and have a '
                             f'non-None wgcna_label_file')
        else:
            df = pd.read_csv(wgcna_label_file, index_col=0)
            # Permute labels vs genes
            df['gene_id'] = np.random.permutation(df['gene_id'])
            # Save the DataFrame to a temporary file
            with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv",
                                             delete=True) as temp_file:
                df.to_csv(temp_file.name, index=False)
                self.assign_clusters_from_wgcna(temp_file)

    def save_random_modules_for_goa_find_enrichment(self, wgcna_label_file, out_dir: Path):
        self.do_random_clustering_with_given_size_dist(
            wgcna_label_file=wgcna_label_file,
        )
        assert self.has_been_clustered
        for module_nr, genes in self.get_genes_per_cluster().items():
            file_path = out_dir / f'random_dists_ds1_module_{module_nr}.csv'
            df = pd.DataFrame.from_dict({'gene_id': genes})
            df.to_csv(file_path, index=False, header=False)
        # Undo clustering
        self.df = self.df.drop('cluster_id', axis=1)
        self.has_been_clustered = False

    def assign_clusters_from_split_by_module_files(self, files: list):
        """From a list of files generated by split_by_module
        (i.e. each cluster in a separate file, go over them and use them to
        cluster the genes in this object again

        :param files: list of file paths that should be used to cluster objects.
        Should have filenames such as  atted_dists_wgcna_clustered_ds1_module_0.csv
        """
        assert not self.has_been_clustered
        # self.df['cluster_id'] = np.nan
        assert len(files) > 0, 'No files found'
        df_list = []
        for file in files:
            module_id = int(file.stem.split('module_')[1])
            cluster_df = pd.read_csv(file, header=None, index_col=0)
            # Assign cluster label to these genes
            cluster_df['cluster_id'] = module_id
            df_list.append(cluster_df)
        mapping_df = pd.concat(df_list)

        self._apply_cluster_mapping_from_df(mapping_df)
        # Check that each gene has been assigned a cluster

    def do_z_scaling(self):
        # Correct axis
        if self.has_been_clustered:
            clusters = self.df['cluster_id']
            self.df = self.df.drop('cluster_id', axis=1)

        self.df = self.df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        # self.df = self.df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

        assert np.all(self.df.mean(axis=1) < 0.01)

        if self.has_been_clustered:
            self.df['cluster_id'] = clusters
        self.has_been_scaled = True

