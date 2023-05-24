import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.spatial.distance import euclidean
from lmfit import Parameters
from matplotlib import pyplot as plt
from scipy.integrate._ivp.ivp import OdeResult
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.interpolate import PchipInterpolator

def standardize(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """Normalize gene expression data

    Based on https://github.com/saeyslab/moduledetection-evaluation/blob/master/lib/methods/clustering.py
    """
    logging.warning("Not sure if we want to do normalisation row-wise, column-wise, or both ways. Make sure you don't forget this")
    # array = df.to_numpy()
    # norm_array = (array - array.mean()) / (array.std())
    # df.iloc[:, :] = norm_array
    #
    # return df
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
    for i, row in enumerate(y_real):
        ax1.plot(t_real, row, label=f'Module{i}')
    ax1.set_title('y')
    ax1.set_ylabel('Gene expression')
    ax1.set_xlabel('Time (h)')
    ax1.legend()

    if model_fit:
        for row in model_fit.y:
            ax2.plot(model_fit.t, row)
        ax2.set_title('y_hat')
        ax2.set_ylabel('Gene expression')
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


def calculate_coefficient_of_variation(x: np.ndarray):
    """Calculate coefficient of variation for input array of numbers"""
    return np.std(x, ddof=1) / np.mean(x)


def calculate_qcd(x: np.ndarray):
    """Calculate quartile coefficient of dispersion for input array of numbers"""
    q1, q3 = np.quantile(x, [.25, .75])
    return (q3 - q1) / (q3 + q1)


def extract_flor_id_genes(flor_id_html_path: Path,
                          out_path: Path):
    """Take (slightly preprocessed) html of FlorID flowering genes
    and extract all florid genes from here
    """
    html_text = flor_id_html_path.read_text()
    dfs = pd.read_html(html_text)
    df = dfs[0]
    df.columns = ['name', 'short_name', 'keyword', 'effect_on_flowering',
                  'conditions_for_effect', 'phenotype', 'locustag', 'appears_in',
                  'key_articles']
    df.to_pickle(out_path)


def calculate_parameter_distance(guessed_params: Parameters,
                                 true_params: Parameters) -> float:
    """For two sets of parameters, calculate their euclidean distance

    :param guessed_params: Parameters that the fit found
    :param true_params: Parameters that are the ground trugh
    :return: Euclidean distance between the two sets of parameters
    """
    param_array1 = [v.value for v in guessed_params.values()]
    param_array2 = [v.value for v in true_params.values()]
    return euclidean(param_array1, param_array2)


def do_pca(df: pd.DataFrame) -> np.ndarray:
    """Represent a module as the value of its first principal component.

    Instead of a mean value of all genes. The approach we do here is comparable
    to an eigengene.

    :param df: Input dataframe that contains column 'cluster_id' to indicicate the
    cluster number.
    :return: pca value of first principal component for all conditions"""
    # Is this check really needed? Not sure...
    if df is not None:
        cluster_id = df['cluster_id'].iloc[0]
        df = df.drop('cluster_id', axis=1)
        pca = PCA(n_components=1)
        pca_values = pca.fit_transform(df.T)
        explained_var = pca.explained_variance_ratio_[0]
        assert explained_var > .4, "First PC does not explain >40% of the variance"

        # assert np.corrcoef(df.mean().to_numpy(), pca_values.T)[0, 1] > .2, 'Correlation between mean and first PC is too low?'
        plt.plot(df.mean().to_numpy(), pca_values, 'o', label=cluster_id)
        plt.xlabel('Mean expression')
        plt.ylabel(f'PC1 ({explained_var*100:.2f}%)')
        # plt.show()
        return pca_values