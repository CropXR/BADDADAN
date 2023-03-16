import logging
import re
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate._ivp.ivp import OdeResult
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.interpolate import make_interp_spline, BSpline, PchipInterpolator


def standardize(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """Normalize gene expression data

    Based on https://github.com/saeyslab/moduledetection-evaluation/blob/master/lib/methods/clustering.py
    """
    # I assume this normalizes per sample in the original implementation?
    if axis == 0:
        return (df - df.mean()) / df.std()
    elif axis == 1:
        transposed_df = df.T
        row_normalised = (transposed_df - transposed_df.mean()) / (transposed_df.std())
        return row_normalised.T

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


def plot_y_and_y_hat(y_real: np.ndarray, t_real: np.ndarray | list,
                     model_fit: OdeResult = None):
    """Plot real data y and model prediction y_hat

    :param y_real: Numpy array containing measured data. One variable per row
    :param model_fit: Numpy array, must be same shape as y
    :param t_real: Time points at which to draw the function. Must be same for
               both arrays
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all')
    # fig.set_size_inches(15, 8)
    # fig.suptitle('Comparison real vs estimated')
    for i, row in enumerate(y_real, start=1):
        ax1.plot(t_real, row, label=f'Module{i}')
    ax1.set_title('y')
    ax1.set_ylabel('Log2 gene expression')
    ax1.set_xlabel('Time (h)')
    ax1.legend()

    if model_fit:
        for row in model_fit.y:
            ax2.plot(model_fit.t, row)
        ax2.set_title('y_hat')
        ax2.set_ylabel('Log2 gene expression')
        ax2.set_xlabel('Time (h)')

    plt.show()

def fit_spline(data: np.ndarray, time: list | np.ndarray,
               num_timepoints: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate through data

    :param data: numpy array, which should contain columns that correspond
    to observations at different time points, and row that correspond to
    different variables (e.g. gene modules)
    :param time: Time points at which original data was measured
    :param num_timepoints: How many timepoints to interpolate
    :return: new_timepoints, and interpolated data that belongs to it.
    """
    new_time = np.linspace(min(time), max(time), num=num_timepoints)
    out_array = np.empty((len(data), num_timepoints))
    for i, row in enumerate(data):
        # spline = make_interp_spline(time, row)
        spline = PchipInterpolator(time, row)
        out_array[i, :] = spline(new_time)
    return new_time, out_array
