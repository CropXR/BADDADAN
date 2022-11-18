import re
from pathlib import Path

import GEOparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from ExpressionMatrix import ExpressionMatrix

pd.options.display.width = 0
GEOparse.logger.set_verbosity('INFO')


def split_based_on_temp(expression_matrix: ExpressionMatrix,
                        n_splits: int = 3,
                        n_repeats: int = 3):
    """Function which takes expression matrix, and splits it into stratified
    k-fold CV set, stratified for each temperature.

    :param expression_matrix: ExpressionMatrix which should be split
    :param n_splits: into how many folds to split the ExpressionMatrix. Default: 3.
    :param n_repeats: How many times to repeat the
    :return: Generator containing ExpressionMatrixTrain and ExpressionMatrixTest
    """
    sample_names = expression_matrix.get_column_names()
    my_regex = re.compile(r'\d+')
    temp_per_sample = [re.search(my_regex, sample).group()
                       for sample in sample_names]
    k_fold_splitter = RepeatedStratifiedKFold(n_splits=n_splits,
                                              n_repeats=n_repeats)
    for train_idx, test_idx in k_fold_splitter.split(sample_names, temp_per_sample):
        train_set_cols, test_set_cols = (sample_names[train_idx],
                                         sample_names[test_idx])
        yield expression_matrix.extract_train_test(train_set_cols, test_set_cols)


def parse_geo_file(file_path: Path):
    """From a file path, correctly parse GEO expression file and
    return ExpressionMatrix object.

    :param file_path: path to GEO expression file. Works on .soft format,
                      others have not been tested.
    """
    gse = GEOparse.get_GEO(filepath=str(file_path), silent=True)
    # Merge all samples into one dataframe
    df = gse.pivot_samples("VALUE")
    # Convert sample names to titles that humans can understand
    better_name_dict = gse.phenotype_data.title.to_dict()
    df.columns = [better_name_dict[old_name] for old_name in df.columns]
    return ExpressionMatrix(df)


def wrapper(file_path, n_cluster):
    y_true_list = []
    y_pred_list = []
    reg_scores = []
    expression_matrix_temp = parse_geo_file(file_path)
    expression_matrix = expression_matrix_temp.remove_non_wt()

    for expr_matr_train, expr_matr_test in split_based_on_temp(expression_matrix):
        de_genes = expr_matr_train.get_only_de_genes()
        overview_df = de_genes.extract_modules(n_cluster)
        gene_to_module = de_genes.get_cluster_per_gene()

        # Y values inferred from paper
        # 23 degrees -> 15 leaves
        # 16 degrees -> 30 leaves
        x_train = overview_df.to_numpy()
        y_train = overview_df.index.map(lambda x: 30 if '16' in x else 15).to_numpy()

        reg = LinearRegression()
        reg.fit(x_train, y_train)

        # Do inference
        test_overview = expr_matr_test.expressions_of_predefined_clusters(
            gene_to_module)
        x_test = test_overview.to_numpy()
        y_true = test_overview.index.map(lambda x: 30 if '16' in x else 15).to_numpy()
        y_pred = reg.predict(x_test)
        r_squared = reg.score(x_test, y_true)
        reg_scores.append(r_squared)
        y_pred_list.extend(y_pred)
        y_true_list.extend(y_true)
    print('Mean r squared')
    print(np.mean(reg_scores))
    return y_pred_list, y_true_list, reg_scores


def do_cv_for_nclust(file_path):
    """Do cross-validation to determine optimal number of clusters"""
    max_clust = 20
    cluster_numbers = [i for i in range(1, max_clust)]
    r_squared_values = []
    for cluster_number in cluster_numbers:
        _, _, r_squared = wrapper(file_path, cluster_number)
        r_squared_values.append(r_squared)
    mean_rsquared = [np.mean(r_sq) for r_sq in r_squared_values]
    plt.errorbar(cluster_numbers, mean_rsquared,
                 yerr=[np.std(r_sq) / np.sqrt(len(r_sq)) for r_sq in
                       r_squared_values],
                 linestyle='', color='black', capsize=4)
    plt.plot(cluster_numbers, mean_rsquared, 'o-')
    plt.xticks(np.arange(0, max_clust, 2))
    plt.xlabel('Nr clusters')
    plt.ylabel('R squared')
    plt.show()

def plot_pred_vs_real(file_path, n_cluster, k):
    y_pred_list, y_true_list, _ = wrapper(file_path, n_cluster)
    plt.plot(y_true_list, y_pred_list, 'o')
    x_ref = np.linspace(10, 30)
    y_ref = x_ref
    plt.plot(x_ref, y_ref)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.show()

def main(
        file_path: Path,
        n_cluster: int = typer.Option(5, help='Number of gene clusters to '
                                              'extract'),
        n_repeats: int = typer.Option(15, help='Number of times that k-fold cv should be repeated')
):
    do_cv_for_nclust(file_path)
    # plot_pred_vs_real(file_path, n_cluster, n_repeats)


if __name__ == "__main__":
    typer.run(main)
