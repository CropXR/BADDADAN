import logging
import re
import string
from pathlib import Path
from typing import List, Iterable

import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import bootstrap
from matplotlib import pyplot as plt

from scipy.integrate._ivp.ivp import OdeResult
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.interpolate import PchipInterpolator


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


def get_info_from_gse65046(sample_names: list[str] | pd.Index | pd.DataFrame) -> dict:
    """From sample names that are used in GSE65046, extract time, condition,
     replicate number.

    Example sample name: '3 control b'

    :param sample_names: List of sample names
    :return: out dict with keys: time, condition, and rep_nr.
    """
    out_dict = {'time': [],
                'condition': [],
                'rep_nr': []}
    for sample in sample_names:
        time, condition, rep_letter = sample.split(' ')
        time = pd.to_timedelta(time + ' days')
        out_dict['time'].append(time)
        out_dict['condition'].append(condition)
        rep_nr = string.ascii_lowercase.index(rep_letter)
        out_dict['rep_nr'].append(rep_nr)
    return out_dict


def get_info_from_gse5628(sample_names: list[str] | pd.Index | pd.DataFrame) -> dict:
    """From sample names that are used in GSE5628, extract time, tissue and replicate number.
    Example sample name: 'AtGen_6-9411_Heatstress(3h)+3hrecovery-Shoots-6.0h_Rep1'

    :param sample_names: List of sample names
    :return: out dict with keys: time, tissue, and rep_nr.
    """
    out_dict = {'time': [],
                'tissue': [],
                'rep_nr': []}
    for sample in sample_names:
        # time = re.search(r'\d+\.\d+h', sample).group()
        time = re.search(r'(\d+\.)?\d+h_', sample).group()
        time = pd.to_timedelta(time[:-1])
        out_dict['time'].append(time)
        tissue = re.search(r'Shoots|Roots', sample).group()
        out_dict['tissue'].append(tissue)
        rep_nr = re.search(r'Rep\d', sample).group()
        out_dict['rep_nr'].append(rep_nr)
    return out_dict

def get_info_from_emtab375(sample_names):
    """Handle sample names from the EMTAB375 samples"""
    out_dict = {'time': [],
                'condition': [],
                'light': []}
    for sample in sample_names:
        if ';' in sample:
            # handle edge case (i.e. the first sample)
            time = '0'
            temp = '21'
            light = 'normal light (150 uE)'
        else:
            time = re.search(r'\d+$', sample).group()
            temp = re.search(r'^\d+', sample).group()
            # Get the middle bit
            light = re.search(r'(?<=\d\s).+(?=\s\d+$)', sample).group()
        time = pd.to_timedelta(time + ' minutes')
        out_dict['time'].append(time)
        out_dict['condition'].append(temp)
        out_dict['light'].append(light)
    return out_dict


def plot_y_and_y_hat(y_real: np.ndarray, t_real: np.ndarray | list,
                     model_fit: OdeResult = None, axs = None, data_point_overlay=False,
                     error_bars: np.ndarray | list = None):
    """Plot real data y and model prediction y_hat

    :param y_real: Numpy array containing measured data. One variable per row
    :param model_fit: Numpy array, must be same shape as y
    :param t_real: Time points at which to draw the function. Must be same for
               both arrays
    :return:
    """

    if axs is None:
        _, (ax1, ax2) = plt.subplots(1, 2, sharey='all')
    elif data_point_overlay:
        ax1 = ax2 = axs
    else:
        ax1, ax2 = axs
    # fig.set_size_inches(15, 8)
    # fig.suptitle('Comparison real vs estimated')
    line_type = '.' if data_point_overlay else '-'
    colours = sns.color_palette(n_colors=len(y_real))
    for i, row in enumerate(y_real):
        if error_bars is not None:
            selected_errors = error_bars.iloc[i, :].tolist()
            ax1.errorbar(t_real, row, yerr=selected_errors,
                         fmt=line_type, label=f'Module{i}', color=colours[i])
        else:
            ax1.plot(t_real, row, line_type, label=f'Module{i}', color=colours[i])
    ax1.set_title('y')
    ax1.set_ylabel('Gene expression')
    ax1.set_xlabel('Time (h)')
    ax1.legend()

    if model_fit:
        if not data_point_overlay:
            ax2.set_title('y_hat')
            ax2.set_ylabel('Gene expression')
            ax2.set_xlabel('Time (h)')
        for i, row in enumerate(model_fit.y):
            ax2.plot(model_fit.t, row, color=colours[i])
        # ax2.get_legend().remove()


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

def check_all_identical_lists(lists: List[List]) -> bool:
    """Check if all sublists in a list of lists are identical.

    :param lists: A list of lists to be checked.

    :return: True if all sublists are identical, False otherwise.
    """
    if len(lists) == 0:
        return False  # Empty list, not all identical

    first_sublst = lists[0]
    for sublst in lists[1:]:
        if sublst != first_sublst:
            # Found a different sublist, not all identical
            return False
    # All sublists are identical
    return True


def mean_bootstrap_error(in_df: pd.DataFrame, confidence_level: float = .95) -> pd.Series:
    """From a dataframe, calculate the per-column mean bootstrap error"""
    in_df = in_df.drop('cluster_id', axis=1)
    x = in_df.to_numpy()
    all_bs = bootstrap((x,), np.mean,
                       confidence_level=confidence_level).confidence_interval
    bs_error = (all_bs.high - all_bs.low) / 2
    return pd.Series(bs_error)

def one_gene_list_file_per_cluster(in_dir: Path,
                                   out_dir: Path,
                                   use_for_analysis_func: callable):
    """
    
    :param in_dir: Directory that contains files of clustered dataset
    :param out_dir: Directory to save each module as seperate file (needed for GO enrichment)
    :param use_for_analysis_func: Takes file name as input, and returns bool to indicate if file should be processed. If true the file is processed.
    Used to select e.g. only certain methods or deepsplit values for analysis
    :return: 
    """
    out_dir.mkdir(exist_ok=True)
    for file in in_dir.iterdir():
        if not use_for_analysis_func(file.name):
            logging.info(f'Skipping {file.name}')
            continue
        logging.info(f'Processing {file.name}')
        df = pd.read_csv(file, index_col=0)
        for module_name, group_df in df.groupby('colors'):
            out_file_name = f'{file.stem}_module_{module_name}.csv'
            group_df['gene_id'].to_csv(
                out_dir / out_file_name, index=False, header=False)

def keep_common_genes_in_dfs(df1, df2):
    # get intersection
    selected_genes = df1.index.intersection(df2.index)
    df1 = df1.loc[selected_genes, selected_genes]
    df2 = df2.loc[selected_genes, selected_genes]
    return df1, df2, selected_genes
