"""
Contains classes that contain matrices of gene expression levels.
"""
from __future__ import annotations
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Callable, Dict
import re
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns
import GEOparse
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import qnorm
from scipy.integrate._ivp.ivp import OdeResult
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import svd

from Expressions.ExpressionArrayAnnotation import ExpressionArrayAnnotation
from helpers import get_info_from_gse5628, standardize, \
    calculate_coefficient_of_variation, calculate_qcd, do_pca


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
        self.df.index = self.df.index.str.upper()
        # Drop duplicates
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.has_been_clustered = False
        self.column_parser: Callable = None
        # TODO declare this from the start?
        self.phenotype_dict = None

    def __repr__(self):
        return (f'ExpressionMatrix with {len(self.df)} genes'
                f' and columns {self.df.columns.to_list()}')

    @classmethod
    def from_csv(cls, file_path: Path, log2_transform: bool = False,
                 sep: str = ','):
        df = pd.read_csv(file_path, sep=sep, index_col=0)
        if log2_transform:
            df = np.log2(df)
        return cls(df)

    @classmethod
    def from_geo_file(cls, file_path: Path,
                      array_annotation: ExpressionArrayAnnotation = None,
                      annotate_from_gpl: bool = False,
                      log2_transform: bool = False,
                      name_to_drop: str = None):
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
        gse = GEOparse.get_GEO(filepath=str(file_path), silent=True)
        # Merge all samples into one dataframe
        df = gse.pivot_samples("VALUE")
        # Probe ID must be uppercase
        df.index = df.index.str.upper()
        if name_to_drop:
            logging.info(f'Dropping all probes that contain {name_to_drop}')
            df = df.loc[df.index.map(lambda x: name_to_drop not in x), :]
        assert not (annotate_from_gpl and array_annotation), \
            "Only provide one way of converting from probe ID to gene names"

        if annotate_from_gpl:
            assert len(gse.gpls) == 1, "GSE contains more than one platform?"
            # Take first (and only) value from dict
            gpl_object = sorted(gse.gpls.values())[0]
            gpl_table = gpl_object.table
            # Probe ID must be uppercase
            gpl_table['ID'] = gpl_table['ID'].str.upper()
            df = df.merge(gpl_table[['ID', 'ORF']], left_index=True, right_on='ID')
            # Drop NA mappings
            logging.info(f'{len(df)} Probe IDs at start')
            df = df.dropna(subset='ORF')
            logging.info(f'{len(df)} genes mapped to probe IDs')
            df = df.set_index('ORF')
            # Remove old probe ID column
            df = df.drop('ID', axis=1)

        elif array_annotation:
            # Get gene names based on ExpressionAnnotation object
            logging.info('Converting probe names to genes...')
            new_indices = df.index.map(array_annotation.probe_to_agi)
            # Count how many probe names were not mapped to a gene by the
            # annotation file, i.e. their name did not change.
            unmapped_probes = new_indices.intersection(df.index)

            logging.info(
                f'Could not find annotation of {len(unmapped_probes)} probes '
                f'({len(unmapped_probes) / len(df.index):.2%}). '
                f'Proceeding with their original names')
            if len(unmapped_probes) < 10:
                for probe in unmapped_probes:
                    logging.info(probe)
            df.index = new_indices

        if log2_transform:
            df = np.log2(df)
        # Convert sample names to titles that humans can understand
        better_name_dict = gse.phenotype_data.title.to_dict()
        df.columns = [better_name_dict[old_col] for old_col in df.columns]
        return cls(df)

    def concat_to_expression_matrix(
            self, new_expression_matrix: ExpressionMatrix,
            keys: list[str]):
        """Concatenates a second expression matrix to this expression matrix
         to enable clustering on multiple experiments.

        :param new_expression_matrix: The expression matrix to be concatenated.
        :param keys: The names to be used for the columns in the
            concatenated expression matrix. I.e. shorthand names for
            the experiments.
        """
        both_dfs = pd.concat([self.df, new_expression_matrix.df], axis=1, keys=keys)
        self.df = both_dfs

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
        else:
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

    def remove_condition_from_expression_matrix(self, key: str):
        """Removes a specified condition from the expression matrix.

        :param key: The condition to be removed.
        :raises AssertionError: If the expression matrix does not have
                                multiple levels in the columns.
        """
        assert self.df.columns.nlevels > 1, """Seems there are not multiple 
            levels in the columns. I.e. not multiple conditions"""

        # Columns where top index matches
        columns_to_drop = self.df.columns[self.df.columns.get_level_values(0) == key]
        self.df = self.df.drop(columns=columns_to_drop, axis=1)

        # Temporarily remove cluster_id
        cluster_id = self.df.cluster_id
        self.df = self.df.drop(columns='cluster_id', axis=1)

        # If only one top level is left, just go to normal index again
        if len(set(self.df.columns.get_level_values(0))) == 1:
            self.df = self.df.droplevel(0, axis=1)
        # Re-add cluster_id
        self.df['cluster_id'] = cluster_id

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

    def plot_per_gene_std(self) -> None:
        """Plot per gene standard deviation across samples"""
        sns.histplot(self.df.std(axis=1))
        plt.show()

    def plot_sample_gene_heatmap(self) -> None:
        sns.clustermap(self.df, method='complete', metric='correlation',
                       yticklabels=False)
        plt.show()

    def _calculate_gene_variation(self, method: Literal['std', 'mad', 'cv', 'qcd']) -> pd.Series:
        """Calculate for each gene how much it varies over all the samples

        :param method: How to determine variation between samples:
                        std: standard deviation
                        mad: mean absolute deviation
                        qcd: quartile coefficient of dispersion
        :return: Measure of variation for each gene
        """

        if method in ['qcd', 'cv']:
            # Cannot take negative values:
            # Subtract minimum value to all values, so qcd / cv does not
            # get negative inputs
            min_value = self.df.min().min()
            # all_num_df = self.df.apply(pd.to_numeric)
            translated_df = self.df - min_value

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
            case 'std':
                return self.df.std(axis=1)
            case 'mad':
                return self.df.mad(axis=1)
            case 'qcd':
                return translated_df.apply(calculate_qcd, axis=1)
            case 'cv':
                return translated_df.apply(calculate_coefficient_of_variation, axis=1)
            case _:
                raise NotImplementedError

    def keep_genes_above_deviation_cutoff(
            self, cutoff: float = None,
            method: Literal['std', 'mad', 'qcd'] = 'std'
    ) -> None:
        """Remove non-differentially expressed (de) genes.

        :param method: How to determine variation between samples:
                        std: standard deviation
                        mad: mean absolute deviation
                        qcd: quantile coefficient of dispersion
        :param cutoff: Minimum standard deviation between samples
                           for a gene to be included. Default: 1.0
        """
        variation = self._calculate_gene_variation(method)
        # Only keep genes that are above the cutoff
        self.df = self.df[variation > cutoff]

    def keep_n_most_deviating_genes(self, n_max: int = None,
                                    method: Literal['std', 'mad', 'cv', 'qcd'] = 'std') -> None:
        """Remove non-differentially expressed (de) genes.

        :param method: How to determine variation between samples:
                        std: standard deviation
                        mad: mean absolute deviation
                        cv: coefficient of variation
        :param n_max: Number of genes to keep. E.g. 1000 will give you the
            top 1000 genes with highest variation between samples.
        """
        variation = self._calculate_gene_variation(method)
        subset_genes = variation.sort_values(ascending=False).head(n_max)
        self.df = self.df.loc[subset_genes.index]

    def quantile_normalize(self, ref_mappings: ExpressionMatrix | None = None):
        if ref_mappings is None:
            # Get rid of duplicates
            self.df = self.df[~self.df.index.duplicated(keep='first')]
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

    def save_expression_to_txt(self, file_name: Path, do_standardize=True):
        """Save the expression values to txt.

        File is formatted so the expressions
        can be analysed by FLAME (https://code.google.com/archive/p/flame-clustering/source/default/source).
        I.e. the first row describes the number of rows and columns, and
        items in every row are seperated by a space.

        :parameter file_name: Path on which to save the txt output file
        """
        nr_rows, nr_cols = self.df.shape
        first_line = f'{nr_rows} {nr_cols}\n'

        with file_name.open('w+') as f:
            f.write(first_line)

        df = self.df.copy()
        # Save all the gene names
        gene_names_with_index = df.index.to_series(
            index=[i for i in range(len(df))])

        # Standardize (if needed) and save the dataframe
        normalized_df = standardize(df, axis=1) if do_standardize else df
        normalized_df.to_csv(file_name, sep=' ', header=False, index=False,
                             mode='a')
        return gene_names_with_index.to_dict()

    def assign_clusters_based_on_already_clustered_expr_mat(
            self, clustered_expressions: ExpressionMatrixTimeSeries) -> None:
        """Assign clustering to this expression matrix from another
        ExpressionMatrix where the genes have already been clustered.

        Also drops all genes from this df that are not mentioned in
        the clustering.

        :param clustered_expressions: ExpressionMatrix which contains
        clustering of genes.
        """
        assert (not self.has_been_clustered
                and clustered_expressions.has_been_clustered)
        assert 'cluster_id' not in self.df.columns
        # TODO speedups here?
        assigned_modules = self.df.index.map(
            lambda x: clustered_expressions.get_cluster_per_gene().get(x))
        inference_df = self.df.assign(cluster_id=assigned_modules)
        self.df = inference_df.dropna(subset='cluster_id')
        self.has_been_clustered = True
        assert (self.get_gene_names().sort()
                == clustered_expressions.get_gene_names().sort())

    def assign_clusters_from_tf2_input(self, tf2_input_path: Path):
        """Find which genes belong to which cluster, based
        on the TF2Network input file.

        :param tf2_input_path: Path to tf2network input file, should
                                contain lines with cluster_id and gene
                                name separated by space.
        """
        print()
        mapping_df = pd.read_csv(tf2_input_path, sep=' ', names=['cluster_id','gene_name'], index_col='gene_name')
        self._apply_cluster_mapping_from_df(mapping_df)

    def assign_clusters_from_jordi_input(self, input_file_jordi, drop_duplicates=False):
        """

        :param input_file_jordi:
        :return:
        """
        df = pd.read_csv(input_file_jordi)
        sub_df_list = []
        # Take every second column from the dataframe
        for i in range(0, df.shape[1], 2):
            series = df.iloc[:, i]
            module_id = int(re.search(r'_\d_', series.name).group()[1])
            sub_df = series.to_frame(name='gene_name')
            # Make all gene names capitalized
            sub_df['gene_name'] = sub_df['gene_name'].str.capitalize()
            sub_df['cluster_id'] = module_id
            sub_df_list.append(sub_df)

        mapping_df = pd.concat(sub_df_list, axis=0)
        mapping_df = mapping_df.set_index('gene_name')
        if drop_duplicates:
            mapping_df = mapping_df[~mapping_df.index.duplicated(keep='first')]
        self._apply_cluster_mapping_from_df(mapping_df)

    def _apply_cluster_mapping_from_df(self, mapping_df):
        """

        :param mapping_df:
        :return:
        """
        assert not mapping_df.index.has_duplicates, 'Mapping dataframe contains duplicate indices'
        # TODO is this proper way to handle mismatch?
        self.df.index = [i.split(';')[0] for i in self.df.index]
        # Drop duplicates based on index
        self.df = self.df.reset_index().drop_duplicates(
            subset='index').set_index('index')
        new_df = pd.concat([self.df, mapping_df], join='inner', axis=1)
        logging.info(f'Lost {len(mapping_df)-len(new_df)} genes during annotation')
        self.df = new_df
        self.has_been_clustered = True


class ExpressionMatrixTraining(ExpressionMatrix):
    """Can be created from ExpressionMatrix by command like:

    my_expression_matrix = ExpressionMatrix.from_geo(some_path)
    training_df = ExpressionMatrixTraining(my_expression_matrix.df)

    """
    def extract_module_expressions(self, n_cluster: int,
                                   for_static_predictions = False,
                                   do_plotting: bool = False,
                                   random_clustering: bool = False,
                                   aggregation_method: Literal['mean', 'pca'] = 'mean'
                                   ) -> pd.DataFrame:
        """Get mean expression per gene module based on
        clustering of expression correlation. And return as dataframe.

        :param n_cluster: Number of clusters
        :param random_clustering: If true, assign each module a random cluster
        :param do_plotting: If true, plot the clustering
        :param for_static_predictions: If true, output is formatted so it can
                                        be used with the static prediction
                                        pipeline
        """
        # Make sure clustering has been performed
        if not self.has_been_clustered:
            logging.info('Not clustered yet, performing clustering now')
            if random_clustering:
                self._do_random_clustering(n_cluster)
            else:
                self.do_hierachical_clustering(n_cluster, do_plotting=do_plotting)
        else:
            logging.info('Already clustered, will not perform clustering again')

        if aggregation_method == 'mean':
            # DO MEAN CLUSTERING
            molten_df = pd.melt(self.df, id_vars='cluster_id',
                                value_vars=self.df.columns[:-1],
                                ignore_index=False, var_name='sample',
                                value_name='expression')
            summary_df = molten_df.groupby(
                ['sample', 'cluster_id']).mean().reset_index()
            if for_static_predictions:
                return summary_df.pivot(index='sample', columns='cluster_id',
                                        values='expression')
            return summary_df
        elif aggregation_method == 'pca':
            # # PCA-based
            pca_values = self.df.groupby('cluster_id').apply(do_pca)
            plt.legend()
            plt.show()
            raise NotImplementedError("Cannot yet use these pca values"
                                      " downstream in the pipeline")
        else:
            raise NotImplementedError(f'{aggregation_method=} is not implemented')
        # PERFORM EIGENGENE clustering at each timepoint(?)

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
        # Calculate pearson correlation
        subset_corr = np.corrcoef(self.df)
        # subset_corr = np.abs(subset_corr)
        # subset_corr = subset_corr**2

        # Calculate distance
        dist = 1 - subset_corr
        # Squareform diagonality checking can be too strict, so we do it
        # explicitly here and disable it in the squareform function call
        assert np.allclose(dist, dist.T), 'Matrix does not appear symmetrical?'
        assert sum(np.diag(dist)) < 1e-6, 'Sum of diagonal too high'
        dense_dist = squareform(dist, checks=False)

        # Create linkage matrix and infer clusters
        linkage_matrix = linkage(dense_dist, method='complete')
        # linkage_matrix = linkage(dense_dist, method='average')
        clustering = fcluster(linkage_matrix, n_cluster, 'maxclust')
        # Make clustering be 0-based instead of 1-based
        clustering = clustering - 1
        if do_plotting:
            # Create colours to use in clustermap
            lut = dict(zip([i for i in range(0, n_cluster + 1)],
                           sns.color_palette(n_colors=n_cluster)))
            row_colors = [lut[i] for i in clustering]
            sns.clustermap(dist, row_linkage=linkage_matrix,
                           col_linkage=linkage_matrix , row_colors=row_colors)
            plt.show()
        self.df = self.df.assign(cluster_id=clustering)
        self.has_been_clustered = True

    def _do_random_clustering(self, n_cluster: int = 2) -> None:
        """Assign genes to random clusters. Used for testing the null hypothesis"""
        # n_cluster+1 because numpy upper limit excludes that integer
        self.df['cluster_id'] = np.random.randint(1, n_cluster+1,
                                                  len(self.df))
        self.has_been_clustered = True


class ExpressionMatrixTimeSeries(ExpressionMatrixTraining):
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

    def keep_only_shoot(self, ignore_cluster_id_col=True) -> None:
        """Keep only columns that originate from shoot"""
        if ignore_cluster_id_col:
            col_mask = [col for col in self.df.columns
                        if 'Shoot' in col or col == 'cluster_id']
        else:
            col_mask = [col for col in self.df.columns
                        if 'Shoot' in col]
        self.df = self.df[col_mask]

    def plot_clusters_over_time(self, plot_units: bool = False, title=None) -> None:
        """Plot expression of clusters over time.

        :param plot_units: If true, plot line for each gene individually.
                            If false, plot mean of all genes in a cluster.
        """
        sns.set_theme()
        some_df = self._get_gene_expression_long_form()
        some_df['time (days)'] = some_df['time'].dt.days

        if plot_units:
            # sns.relplot(data=some_df, x='time (days)', y='expression',
            #             kind='line',
            #             hue='replicate', col='cluster_id',
            #             palette=sns.color_palette(),
            #             units='level_0', estimator=None, lw=1, alpha=.2)
            sns.relplot(data=some_df, x='time (days)', y='expression',
                        kind='line',
                        hue='replicate', col='cluster_id',
                        palette=sns.color_palette(),
                        units='ID_REF', estimator=None, lw=1, alpha=.2)
        else:
            sns.lineplot(data=some_df, x='time (days)', y='expression',
                         hue='cluster_id',
                         palette=sns.color_palette(), errorbar='sd')
            # sns.lineplot(data=some_df, x='time', y='expression',
            #              hue='cluster', style='replicate',
            #              palette=sns.color_palette(), errorbar='sd')

        # Unused bit of code to show the indivudual genes with
        # the mean per module overlay
        # my_thingy = True
        # if my_thingy:
        #     nr_genes = 100
        #     # sub df contains only a few entries
        #     subset_genes = some_df.groupby('cluster').sample(nr_genes)['ID_REF'].to_list()
        #
        #     fig, ax = plt.subplots()
        #     sns.lineplot(data=some_df[some_df['ID_REF'].isin(subset_genes)],
        #                  x='time', y='expression',
        #                  hue='cluster', palette=sns.color_palette(),
        #                  units='ID_REF', estimator=None, lw=1, alpha=.2, ax=ax)
        #     sns.lineplot(data=some_df, x='time', y='expression',
        #                  hue='cluster', style='replicate',
        #                  palette=sns.color_palette(), errorbar=None, ax=ax)
        if title:
            plt.title(title)
        plt.show()
        if self.phenotype_dict:
            for key in self.phenotype_dict:
                sns.lineplot(self.phenotype_dict[key])
                plt.show()

    def corr_to_phenotypes(self):
        df = self._get_gene_expression_long_form()
        df = df.groupby(['time', 'cluster']).mean()['expression'].reset_index()
        for key, values in self.phenotype_dict.items():
            y = values
            for i in range(0, max(df['cluster'])+1):
                sub_df = df[df['cluster'] == i]['expression']
                p = np.corrcoef(sub_df, y)[0][1]
                print(f'{i} to {key}: {p}')
    def _get_cluster_expression_long_form(self, n_clusters: int,
                                          aggregation_method: Literal['mean', 'pca'] = 'mean'):
        """Get dataframe which shows expression of clusters over time
        in long-form dataframe
        """
        out_df = self.extract_module_expressions(n_clusters, do_plotting=False,
                                                 aggregation_method=aggregation_method)
        # Get time point info from sample names
        new_cols = self.column_parser(out_df['sample'].to_list())
        out_df = pd.concat([out_df, pd.DataFrame.from_dict(new_cols)], axis=1)
        out_df['elapsed_mins'] = out_df['time'].astype('timedelta64[m]')
        return out_df

    def _get_gene_expression_long_form(self):
        assert self.has_been_clustered, ('Cluster genes first.'
                                         'I.e. call the .do_hierarchical_clustering() method before calling this method.')
        # TODO create private method of this snippet of code is used twice atm
        # Drop final column and add it later
        df_no_cluster_column = self.df.copy()
        df_no_cluster_column = df_no_cluster_column[df_no_cluster_column.columns[:-1]]

        # First extract meaningful information from column names,
        # so we can group by time and merge biological samples

        column_info = self.column_parser(df_no_cluster_column)
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
        if expression_df.columns[0] != 'ID_REF':
            # Change column name to gene identifier again
            expression_df = expression_df.rename(
                columns={expression_df.columns[0]: 'ID_REF'}
            )
        expression_df = expression_df.merge(self.df['cluster_id'],
                                            left_on='ID_REF', right_index=True)
        return expression_df

    def get_lpan_input_modules(self, n_clusters: int) -> pd.DataFrame:
        """For gene modules, get output that can be used to input into the
        Rscript LPAN workflow. (https://github.com/LiLabAtVT/LPANetwork)

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
        assert is_tf.any(), 'No transcription factors in the modules. ' \
                            'Consider increasing the number of genes that you consider'
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

    def get_clusters_expressions_with_time(
            self,
            n_clusters: int,
            aggregation_method: Literal['mean', 'pca'] = 'mean'
    ) -> tuple[np.ndarray, np.ndarray]:
        """For fitting ODEs, get expression of clusters over time.
        First array in tuple indicates time_points in minutes, second array
        indicates module expressions.

        :return: tuple of time points, module expressions
        """
        some_df = self._get_cluster_expression_long_form(
            n_clusters, aggregation_method=aggregation_method)
        # Take mean of all biological replicates
        some_df = some_df.groupby(['cluster_id', 'elapsed_mins']).mean(
            numeric_only=True).reset_index()
        new_df = some_df.pivot(index='cluster_id', columns='elapsed_mins',
                               values='expression')
        time_points = new_df.columns.to_numpy()
        # Convert time to hours
        time_points = time_points / 60
        module_expressions = new_df.to_numpy()
        return time_points, module_expressions

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

    def add_phenotypes(self, in_dict: Dict[str, list[float]]):
        """Provide phenotypes for this expressionmatrix, to see if modules are
        correlated to a phenotype

        :param in_dict: dictionary that maps name of certain phenotype (key)
                        to its measured value at each time point
        """
        self.phenotype_dict = in_dict
        # for i in self.phenotype_dict.values():
        #     assert len(i) == len(self.)

    def get_correlation(self, module_index: int, tf_name: str,
                        plot: bool = False, method: str = 'pearson') -> float:
        """Calculate the correlation between mean expression of a module and a
        transcription factor

        :param module_index: the index of the module (e.g. 1)
        :param tf_name: Name of transcription factor
        :return: Pearson correlation coefficient
        """
        # TODO eventually check if this can be moved higher up in the hierarchy
        assert self.has_been_clustered, 'Cluster object first'
        if tf_name not in self.df.index:
            logging.warning(f'TF ({tf_name}) not present in dataframe')
            return np.nan
        module_expressions = self.df.groupby('cluster_id').mean()
        selected_module_expression = module_expressions.loc[module_index]

        tf_expressions = self.df.loc[tf_name]
        tf_expressions = tf_expressions.drop('cluster_id')
        if plot:
            fig, ax = plt.subplots()
            ax.plot(selected_module_expression.to_numpy(),
                    tf_expressions.to_numpy(), 'o')
            ax.set_xlabel(f'Module {module_index} expression')
            ax.set_ylabel(f'{tf_name} expression')
            plt.show()
        return tf_expressions.corr(selected_module_expression, method=method)


