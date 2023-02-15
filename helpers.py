import logging
import re
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold


def split_based_on_temp(expression_matrix,
                        n_splits: int = 3,
                        n_repeats: int = 3):
    """Function which takes expression matrix, and splits it into stratified
    k-fold CV set, stratified for each temperature.

    :param expression_matrix: ExpressionMatrix which should be split
    :param n_splits: into how many folds to split the ExpressionMatrix. Default: 3.
    :param n_repeats: How many times to repeat the cv split.
    :return: Generator returning tuples of ExpressionMatrixTrain and ExpressionMatrixTest
    """
    sample_names = expression_matrix.get_sample_names()
    my_regex = re.compile(r'\d+')
    temp_per_sample = [re.search(my_regex, sample).group()
                       for sample in sample_names]
    k_fold_splitter = RepeatedStratifiedKFold(n_splits=n_splits,
                                              n_repeats=n_repeats)
    for train_idx, test_idx in k_fold_splitter.split(sample_names, temp_per_sample):
        train_set_cols, test_set_cols = (sample_names[train_idx],
                                         sample_names[test_idx])
        yield expression_matrix.extract_train_test(train_set_cols, test_set_cols)


def get_info_from_gse5628(sample_names: list[str] | pd.Index) -> dict:
    """From sample names that are used in GSE5628, extract time, tissue and replicate number.
    Example sample name: 'AtGen_6-9411_Heatstress(3h)+3hrecovery-Shoots-6.0h_Rep1'

    :param sample_names: List of sample names
    :return: out dict with keys: time, tissue, and rep_nr.
    """
    out_dict = {'time': [],
                'tissue': [],
                'rep_nr': []}
    for sample in sample_names:
        time = re.search(r'\d+\.\d+h', sample).group()
        time = pd.to_timedelta(time)
        out_dict['time'].append(time)
        tissue = re.search(r'Shoots|Roots', sample).group()
        out_dict['tissue'].append(tissue)
        rep_nr = re.search(r'Rep\d', sample).group()
        out_dict['rep_nr'].append(rep_nr)
    return out_dict

def de_print_fun(xk, convergence=None):
    """Logs the convergence value during differential evolution.

    Can be provided as callback keyword during the differential
    evolution function call.
    """
    logging.info(f'Convergence: {convergence*100:.2f}%')


def plot_y_and_y_hat(y: np.ndarray, y_hat: np.ndarray, t: np.ndarray | list):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(15, 8)
    # fig.suptitle('Comparison real vs estimated')
    for row in y:
        ax1.plot(t, row)
        ax1.set_title('y')

    for row in y_hat:
        ax2.plot(t, row)
        ax2.set_title('y_hat')

    plt.show()
    return