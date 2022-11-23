import logging
from pathlib import Path

import GEOparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LinearRegression

from ExpressionArrayAnnotation import ExpressionArrayAnnotation
from ExpressionMatrix import ExpressionMatrix, ExpressionMatrixTraining
from helpers import split_based_on_temp

pd.options.display.width = 0
GEOparse.logger.set_verbosity('INFO')
logging.basicConfig(level=logging.INFO)

def hierarchical_cluster_expression_matrix(expr_mat: ExpressionMatrix,
                                           n_cluster: int):
    de_genes = expr_mat.get_only_de_genes()
    de_genes.do_hierachical_clustering(n_cluster, inplace=True)

    return


def get_gene_module_names(expr_mat: ExpressionMatrix):
    # Convert input to ExpressionMatrixTraining because we do clustering:
    expr_mat = expr_mat.get_only_de_genes()
    expr_mat = ExpressionMatrixTraining(expr_mat.df)
    expr_mat.do_hierachical_clustering(10, inplace=True)
    some_clustering_dict = expr_mat.get_cluster_per_gene()

    return some_clustering_dict


def lin_regress_on_modules(expr_mat: ExpressionMatrix, n_cluster: int):
    """Takes expressionmatrix, splits into train and test, and reports predicted vs actual values"""
    y_true_list = []
    y_pred_list = []
    reg_scores = []
    for expr_matr_train, expr_matr_test in split_based_on_temp(expr_mat):
        de_genes = expr_matr_train.get_only_de_genes()
        overview_df = de_genes.extract_module_expressions(n_cluster)
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
    logging.info(f'Mean r squared: {np.mean(reg_scores)}')
    return y_pred_list, y_true_list, reg_scores


def do_cv_for_nclust(expr_mat: ExpressionMatrix):
    """Do cross-validation to determine optimal number of clusters"""
    max_clust = 20
    cluster_numbers = [i for i in range(1, max_clust)]
    r_squared_values = []
    for cluster_number in cluster_numbers:
        _, _, r_squared = lin_regress_on_modules(expr_mat, cluster_number)
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


def plot_pred_vs_real(expr_mat: ExpressionMatrix, n_cluster: int):
    x_ref = np.linspace(10, 30)
    y_ref = x_ref
    plt.plot(x_ref, y_ref, 'k:')
    y_pred_list, y_true_list, _ = lin_regress_on_modules(expr_mat, n_cluster)
    plt.plot(y_true_list, y_pred_list, 'o')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.show()


def main(
        expression_path: Path = typer.Option(...,
                                             help='Path to geo expression file'),
        annotation_path: Path = typer.Option(...,
                                             help='Path to annotation of micro array'),
        n_cluster: int = typer.Option(5,
                                      help='Number of gene clusters to '
                                           'extract'),
        n_repeats: int = typer.Option(15,
                                      help='Number of times that k-fold cv should be repeated')
):
    expression_annotation = ExpressionArrayAnnotation(annotation_path)
    expression_matrix = ExpressionMatrix.from_geo_file(expression_path,
                                                       expression_annotation)
    expression_matrix = expression_matrix.remove_non_wt()
    plot_pred_vs_real(expression_matrix, n_cluster)
    # do_cv_for_nclust(expression_matrix)


if __name__ == "__main__":
    typer.run(main)
