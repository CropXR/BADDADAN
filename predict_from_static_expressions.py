import logging
import random
from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, \
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from adjustText import adjust_text

from Expressions.ExpressionMatrix import ExpressionMatrix, ExpressionMatrixTest, \
    ExpressionMatrixTraining
from helpers import split_based_on_temp

# TODO this does need to get a major cleanup at some point


def lin_regress_on_modules(expr_mat: ExpressionMatrix, n_cluster: int):
    """Takes expressionmatrix, splits into train and test,
    performs multiple linear regression and reports predicted vs actual values
    """
    y_true_list = []
    y_pred_list = []
    reg_scores = []
    for expr_matr_train, expr_matr_test in split_based_on_temp(expr_mat):
        expr_matr_train.keep_only_de_genes()
        overview_df = expr_matr_train.extract_module_expressions(n_cluster, for_static_predictions=True)
        gene_to_module = expr_matr_train.get_cluster_per_gene()

        # Y values inferred from paper
        # 23 degrees -> 15 leaves
        # 16 degrees -> 30 leaves
        # Assume that first column is sample name
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

def preprocess_expression_matrix_and_cluster(
        expr_mat: ExpressionMatrixTraining, n_cluster: int,
        std_cutoff: float = 1.0, random_clustering: bool = False) -> tuple[dict, pd.DataFrame]:
    """Do some basic preprocessing on expression matrix, after that
    perform clustering
    """
    expr_mat.keep_only_de_genes(std_cutoff=std_cutoff)
    expr_mat.quantile_normalize()
    overview_df = expr_mat.extract_module_expressions(
        n_cluster,
        for_static_predictions=True, random_clustering=random_clustering)

    gene_to_module = expr_mat.get_cluster_per_gene()
    return gene_to_module, overview_df


def fit_log_regress_model(expr_mat: ExpressionMatrixTraining,
                          n_cluster: int,
                          std_cutoff: float = 1.0,
                          random_clustering: bool = False) -> tuple[LogisticRegression, dict]:
    """Fit logistic regression classifier on static plant expression data

    :param expr_mat: Expressionmatrix to which the logistic regression model should fit
    :return: trained model, and gene_to_module mapping which describes for each gene to which module it belongs
    """

    gene_to_module, overview_df = preprocess_expression_matrix_and_cluster(
        expr_mat, n_cluster, std_cutoff, random_clustering)
    x_train = overview_df.to_numpy()
    # 1 if a late flowering plant, 0 if not
    y_train = overview_df.index.map(
        lambda x: 1 if '16' in x else 0).to_numpy()

    reg = LogisticRegression()
    reg.fit(x_train, y_train)

    logging.info(f'{reg.score(x_train, y_train)=}')

    return reg, gene_to_module


def infer_with_log_reg_model(expr_mat: ExpressionMatrixTest,
                             regression_model: LogisticRegression,
                             gene_to_module: dict, do_plotting: bool = True) -> tuple[float, float]:
    """Classify seed data, using a model trained on plant data"""
    design_df = pd.read_csv('../data/static_datasets/seed_design.txt', sep='\t')
    design_df = design_df[design_df['sample'].isin(expr_mat.df.columns)]

    test_overview = expr_mat.expressions_of_predefined_clusters_seed_data(gene_to_module)
    x_test = test_overview.to_numpy()
    # From experimental design dataframe, get temperature at which the
    # experiment was carried out
    experimental_metadata = design_df[design_df['sample'].isin(
        test_overview.index)]
    # If temperature contains 16, classify as low germination, else high germination rate
    experimental_metadata['y_test_true'] = \
        experimental_metadata['temperature'].apply(lambda x:
                                                   0 if '16' in x else 1)
    y_test_true = experimental_metadata['y_test_true'].to_list()
    y_pred_raw = regression_model.predict_proba(x_test)[:, 1]
    y_pred = regression_model.predict(x_test)

    experimental_metadata['y_pred'] = y_pred
    experimental_metadata['y_pred_raw'] = y_pred_raw
    experimental_metadata['DPA'] = experimental_metadata['DPA'].apply(lambda x: int(x[:-3]))
    experimental_metadata['correctly_classified'] = experimental_metadata['y_pred'] == experimental_metadata['y_test_true']

    accuracy_over_time = experimental_metadata.groupby(['temperature', 'DPA'])['correctly_classified'].mean().reset_index()

    if do_plotting:
        sns.relplot(data=accuracy_over_time, row='temperature', x='DPA', y='correctly_classified', marker='s', facet_kws={'sharex': False}, kind='line')

        plt.show()
        ConfusionMatrixDisplay.from_predictions(y_test_true, y_pred)
        plt.show()

        PrecisionRecallDisplay.from_predictions(y_test_true, y_pred_raw)
        plt.show()

        RocCurveDisplay.from_predictions(y_test_true, y_pred_raw)
        plt.show()

    acc = accuracy_score(y_test_true, y_pred)
    roc_score = roc_auc_score(y_test_true, y_pred_raw)
    logging.info(f'Accuracy: {acc}')
    logging.info(f'\n{confusion_matrix(y_test_true, y_pred)=}')
    logging.info(f'\n{roc_score=}')

    return acc, roc_score

def compare_expression_of_clusters_in_seed_vs_plant(
        my_seed_expressions: ExpressionMatrixTest,
        static_expression_two_temps_arabidopsis: ExpressionMatrix,
        n_cluster: int = 2, std_cutoff: float = .3):
    """Get more insight into how module 1 and 2 change expression during
    hot/cold conditions in the seed vs plant tissue
    """
    static_expression_two_temps_arabidopsis = static_expression_two_temps_arabidopsis.to_expressionmatrix_training()
    static_expression_two_temps_arabidopsis.keep_only_wt_samples()
    gene_to_module, plant_module_expressions = preprocess_expression_matrix_and_cluster(
        static_expression_two_temps_arabidopsis, n_cluster, std_cutoff)

    plant_module_expressions['tissue'] = 'plant'
    plant_module_expressions['DPA'] = 'n.a.'

    plant_module_expressions['in_cold'] = plant_module_expressions.index.map(lambda x: 1 if '16' in x else 0)
    seed_module_expressions = my_seed_expressions.expressions_of_predefined_clusters_seed_data(gene_to_module)
    seed_module_expressions['tissue'] = 'seed'

    seed_design_df = pd.read_csv('../data/static_datasets/seed_design.txt', sep='\t', index_col=0)
    seed_design_df = seed_design_df[seed_design_df.index.isin(seed_module_expressions.index)]
    seed_design_df['in_cold'] = seed_design_df['temperature'].apply(lambda x: 1 if '16' in x else 0)

    all_expressions_df = pd.concat([seed_module_expressions, seed_design_df], axis=1)
    all_expressions_df = all_expressions_df[[1.0, 2.0, 'tissue', 'in_cold', 'DPA']]

    all_expressions_df = pd.concat([all_expressions_df, plant_module_expressions], axis=0)
    sns.set_theme()
    sns.scatterplot(data=all_expressions_df, x=1.0, y=2.0, hue='tissue', style='in_cold')
    texts = []
    for line in range(0, all_expressions_df.shape[0]):
        texts.append(
            plt.text(all_expressions_df[1][line],
                     all_expressions_df[2][line], all_expressions_df['DPA'][line],
                     horizontalalignment='left', size='x-small', color='black'))
    # Prevent overlapping texts
    adjust_text(texts, arrowprops={"arrowstyle": "-", 'color': 'k', 'lw': 0.5})
    plt.tight_layout()
    plt.xlabel('Expression module 1')
    plt.ylabel('Expression module 2')
    plt.show()

    # all_expressions_df = all_expressions_df.melt(id_vars=['tissue', 'in_cold'],
    #                                              var_name='module',
    #                                              value_name='expression')
    # print()
    #
    # # sns.lineplot(data=all_expressions_df, y='expression')
    # sns.catplot(data=all_expressions_df, col='module', y='expression', x='in_cold', kind='bar', hue='tissue')
    # plt.ylim(4,8)
    # plt.show()


def compare_expression_distributions(
        my_seed_expressions: ExpressionMatrixTest,
        static_expression_two_temps_arabidopsis: ExpressionMatrix
):

    static_expression_two_temps_arabidopsis.keep_only_wt_samples()
    static_expression_two_temps_arabidopsis = static_expression_two_temps_arabidopsis.to_expressionmatrix_training()
    static_expression_two_temps_arabidopsis.do_hierachical_clustering(2)

    all_data_df = pd.concat([my_seed_expressions.df[['S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22',
       'S23', 'S24', 'S43', 'S44', 'S45', 'S46', 'S47', 'S48', 'S49', 'S50',
       'S51', 'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60']], static_expression_two_temps_arabidopsis.df], axis=1, join='inner')
    all_data_df = all_data_df.melt(id_vars='cluster_id', ignore_index=False,
                                   var_name='sample', value_name='expression')

    fig, ax = plt.subplots()
    g = sns.violinplot(data=all_data_df, x='sample', y='expression',
                   hue='cluster_id', split=True, ax=ax, inner=None)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()


def compare_pca(dfs: list[pd.DataFrame], n_cluster: int = 2):
    # TODO make the preprocessing here quite similar to how I do it when I fit da modellos


    master_df = pd.concat(dfs, axis=1, join='inner')
    pca = PCA(n_cluster)
    new_data = pca.fit_transform(master_df.T)
    new_df = pd.DataFrame(new_data)
    new_df.index = master_df.columns
    new_col_names = [f"Component {i} ({frac_explained*100:.2f}%)"
                     for (i, frac_explained)
                     in enumerate(pca.explained_variance_ratio_)]
    new_df.columns = new_col_names
    new_df['tissue'] = ['Plant' if 'WT' in i else "Seed" for i in new_df.index]
    sns.scatterplot(data=new_df, x=new_col_names[0], y=new_col_names[1], hue='tissue')
    plt.show()
