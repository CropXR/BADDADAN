import re
from pathlib import Path

import GEOparse
from sklearn.model_selection import RepeatedStratifiedKFold

from ExpressionMatrix import ExpressionMatrix


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

