import logging
import re
import string
from pathlib import Path
from typing import List, Iterable

import numpy as np
import seaborn as sns
import pandas as pd
import yaml

from scipy.stats import bootstrap
from matplotlib import pyplot as plt

from scipy.integrate._ivp.ivp import OdeResult
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.interpolate import PchipInterpolator

from Expressions.ExpressionMatrix import AggregationMethod


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


def mean_bootstrap_error(in_df: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    """From a dataframe, calculate the per-column 95% mean bootstrap error"""
    in_df = in_df.drop('cluster_id', axis=1)
    x = in_df.to_numpy()

    all_bs = bootstrap((x,), np.mean,
                       confidence_level=confidence_level).confidence_interval
    out_df = pd.DataFrame([all_bs.low, all_bs.high], columns=in_df.columns, index=['lower', 'upper'])

    return out_df

    # # Some old crap that can be ignored

    # Only
    # bs_error = (all_bs.high - all_bs.low) / 2
    # out_df = out_df - x.mean(axis=0)
    # assert all(out_df.loc[:, 'lower'] < 0) and all(out_df.loc[:, 'lower'] > 0)
    # out_df.loc[:, 'lower'] = out_df.loc[:, 'lower'].abs()

    # Plot genes
    # # Only control or drought
    # test_df = in_df.loc[:,in_df.columns.str.contains('control')]
    # test_df = in_df.loc[:,in_df.columns.str.contains('drought')]
    # for index, row in test_df.iterrows():
    #     plt.plot(row, alpha=.1)
    # plt.plot()
    # plt.plot(test_df.mean())
    #
    # plt.xlabel('Condition')
    # plt.xticks()
    # plt.ylabel('Expression Value')
    # plt.title('Gene Expression Across Conditions')
    # plt.legend(loc='upper right')
    # plt.show()

    # Luckily looks good :)


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


def config_preprocess(experiment_path: Path) -> tuple[dict, dict, dict]:
    config_path = experiment_path / 'config.yaml'
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    data_params = config['data']
    hyper_params = config['hyperparams']
    experiment_params = config['experiment_data']
    agg_method_dict = {'mean': AggregationMethod.MEAN,
                       'eigengene': AggregationMethod.EIGENGENE}
    hyper_params['agg_method'] = agg_method_dict.get(
        hyper_params.get('agg_method')
    )
    return data_params, hyper_params, experiment_params
