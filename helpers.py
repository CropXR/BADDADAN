import json
import logging
import re
import string
from pathlib import Path
from typing import List, Iterable

import numpy as np
import requests
import seaborn as sns
import pandas as pd



import amici
from amici import ReturnDataView, Model
from scipy.spatial.distance import euclidean, squareform
from scipy.stats import bootstrap
from lmfit import Parameters
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

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

def de_print_fun(xk, convergence=None):
    """Logs the convergence value during differential evolution.

    Can be provided as callback keyword during the differential
    evolution function call.
    """
    logging.info(f'Convergence: {convergence*100:.2f}%')


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

def parse_string_input_data(filter_by_de=False):
    aliases_path = "data/raw_data/string_db/3702.protein.aliases.v12.0.txt"
    links_path = "data/raw_data/string_db/3702.protein.links.v12.0.txt"

    # Convert to square matrix of scores


    aliases_df = pd.read_csv(aliases_path, sep='\t')
    aliases_df = aliases_df[aliases_df['source'] == "KEGG_KEGGID_SHORT"]
    aliases_df = aliases_df.drop('source', axis=1)
    aliases_df = aliases_df.set_index('#string_protein_id')
    alias_dict = aliases_df.to_dict()
    links_df = pd.read_csv(links_path, sep=' ')
    links_df.head()
    # logging.info(
    #     sum(links_df['protein2'].isin(aliases_df['#string_protein_id'])) / len(
    #         links_df))

    df_pivot = links_df.pivot(index='protein1', columns='protein2',
                        values='combined_score', )



    df_pivot.index = df_pivot.index.map(
        lambda x: alias_dict['alias'].get(x, x)
    )
    df_pivot.columns = df_pivot.columns.map(
        lambda x: alias_dict['alias'].get(x, x)
    )



    if filter_by_de:
        de_df = pd.read_csv('limma_de_selection/drought_expr_matrix_limma_filtered.csv', index_col=0)

        empty_df = pd.DataFrame(index=de_df.index, columns=de_df.index)
        selected_genes_df, de_df, common_genes = keep_common_genes_in_dfs(
            df_pivot, empty_df
        )
        selected_genes_df = selected_genes_df.fillna(0)
        return selected_genes_df
    else:
        return df_pivot
    # Some additional analysis
    flat_string = squareform(selected_genes_df)
    flat_string = flat_string / max(flat_string)
    flat_string_dist = 1 - flat_string
    sns.histplot(flat_string)
    plt.yscale('log')
    plt.show()

    return flat_string_dist





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

def call_string_db(list_of_genes: list, species:int, method="ppi_enrichment",) -> dict:
    """
    Call stringdb to get the number of nodes, expected number of edges,
    number of edges, and pvalue for a list of genes.
    Requires internet connection.

    Author: Jordi Alonso Esteve

    Parameters
    ----------
    list_of_genes: list
        List of genes to test
    method: str
        What to do, see stringdsb documentation
    species: int
        Species NCBI identifier (ATH: 3702, FLY: 7227)
    Returns
    -------
    out_dict: dict
        Dictionary with the following keys:
            nn: number of nodes
            e_ne: expected number of edges
            ne: number of edges
            pvalue: pvalue of the test
    """
    raise NotImplementedError
    string_api_url = "https://string-db.org/api"
    # remove "unknown" genes
    list_of_genes = [x for x in list_of_genes if x != "unknown ID"]
    params = {
        "identifiers": "%0d".join(list_of_genes),  # your protein list
        "species": species,  # species NCBI identifier **ARABIDOPSIS thaliana**
        "caller_identity": "BADDADAN",  # your app name
    }
    if method == "ppi_enrichment":
        output_format = "tsv-no-header"
        request_url = "/".join([string_api_url, output_format, method])
        try:
            print("Calling stringdb...")
            results = requests.post(request_url, data=params)
            print(results)
        except:
            #time.sleep(1)
            try:
                results = requests.post(request_url, data=params)
            except:
                print("Error in stringdb")
                return {"nn": 0, "e_ne": 0, "ne": 0, "pvalue": np.NAN}
        for line in results.text.strip().split("\n"):
            try:
                nn, ne, a_nd, lcc, e_ne, pvalue = line.split("\t")
            except:
                # import pdb; pdb.set_trace()
                print("Error in stringdb")
                return {"nn": 0, "e_ne": 0, "ne": 0, "pvalue": np.NAN}
        if float(pvalue) == 0.0:
            pvalue = 1e-16
        out_dict = {
            "nn": nn,
            "e_ne": e_ne,
            "ne": ne,
            "pvalue": pvalue,
            "ne_ene": float(ne) / float(e_ne) if float(e_ne) != 0 else 0,
        }
        return out_dict
    elif method == "enrichment":
        output_format = "json"
        request_url = "/".join([string_api_url, output_format, method])
        params = {
        "identifiers": "%0d".join(list_of_genes),  # your protein list
        "species": species,  # species NCBI identifier **ARABIDOPSIS thaliana**
        "caller_identity": "BADDADAN",  # your app name
        }
        response = requests.post(request_url, data=params)
        ##
        ## Read and parse the results
        ##
        data = json.loads(response.text)
        results = {}
        for row in data:
            term = row["term"]
            #preferred_names = ",".join(row["preferredNames"])
            fdr = float(row["fdr"])
            description = row["description"]
            category = row["category"]
            pvalue = float(row["p_value"])
            if fdr < 0.01:
                results[term] = {
                    #"preferred_names": preferred_names,
                    "fdr": fdr,
                    "p_value": pvalue,
                    "description": description,
                    "category": category,
                }
        # If empty retunr one just with the column names
        if len(results) == 0:
            results["None_found"] = {
                #"preferred_names": "None found",
                "fdr": 0,
                "p_value": 0,
                "description": "None found",
                "category": "None found",
            }
            # make a df
        results = pd.DataFrame(results).T
        # order by pvalue
        results = results.sort_values(by="fdr", ascending=True)
        return results


def keep_common_genes_in_dfs(df1, df2):
    # get intersection
    selected_genes = df1.index.intersection(df2.index)
    df1 = df1.loc[selected_genes, selected_genes]
    df2 = df2.loc[selected_genes, selected_genes]
    return df1, df2, selected_genes

#
# def plot_state_trajectories(
#     rdata: ReturnDataView,
#     state_indices: Sequence[int] | None = None,
#     ax: Axes | None = None,
#     model: Model = None,
#     prefer_names: bool = True,
#     marker=None,
# ) -> None:
#     """
#     Plot state trajectories.
#
#     :param rdata:
#         AMICI simulation results as returned by
#         :func:`amici.amici.runAmiciSimulation`.
#     :param state_indices:
#         Indices of state variables for which trajectories are to be plotted.
#     :param ax:
#         :class:`matplotlib.pyplot.Axes` instance to plot into.
#     :param model:
#         The model *rdata* was generated from.
#     :param prefer_names:
#         Whether state names should be preferred over IDs, if available.
#     :param marker:
#         Point marker for plotting (see
#         `matplotlib documentation <https://matplotlib.org/stable/api/markers_api.html>`_).
#     """
#     if not ax:
#         fig, ax = plt.subplots()
#     if not state_indices:
#         state_indices = range(rdata["x"].shape[1])
#
#     if marker is None:
#         # Show marker if only one time point is available,
#         #  otherwise nothing will be shown
#         marker = "o" if len(rdata.t) == 1 else None
#
#     if model is None and rdata.ptr.state_ids is None:
#         labels = [f"$x_{{{ix}}}$" for ix in state_indices]
#     elif model is not None and prefer_names:
#         labels = np.asarray(model.getStateNames())[list(state_indices)]
#         labels = [
#             l if l else model.getStateIds()[ix] for ix, l in enumerate(labels)
#         ]
#     elif model is not None:
#         labels = np.asarray(model.getStateIds())[list(state_indices)]
#     else:
#         labels = np.asarray(rdata.ptr.state_ids)[list(state_indices)]
#
#     for ix, label in zip(state_indices, labels, strict=True):
#         ax.plot(rdata["t"], rdata["x"][:, ix], marker=marker, label=label)
#
#     ax.set_xlabel("$t$")
#     ax.set_ylabel("$x(t)$")
#     ax.legend()
#     ax.set_title("State trajectories")
#
#
# def plot_observable_trajectories(
#     rdata: ReturnDataView,
#     observable_indices: Iterable[int] | None = None,
#     ax: Axes | None = None,
#     model: Model = None,
#     prefer_names: bool = True,
#     marker=None,
#     edata: amici.ExpData | amici.ExpDataView = None,
# ) -> None:
#     """
#     Plot observable trajectories.
#
#     :param rdata:
#         AMICI simulation results as returned by
#         :func:`amici.amici.runAmiciSimulation`.
#     :param observable_indices:
#         Indices of observables for which trajectories are to be plotted.
#     :param ax:
#         :class:`matplotlib.pyplot.Axes` instance to plot into.
#     :param model:
#         The model *rdata* was generated from.
#     :param prefer_names:
#         Whether observable names should be preferred over IDs, if available.
#     :param marker:
#         Point marker for plotting (see
#         `matplotlib documentation <https://matplotlib.org/stable/api/markers_api.html>`_).
#     :param edata:
#         Experimental data to be plotted (no event observables yet).
#     """
#     if isinstance(edata, amici.amici.ExpData):
#         edata = amici.ExpDataView(edata)
#
#     if not ax:
#         fig, ax = plt.subplots()
#     if not observable_indices:
#         observable_indices = range(rdata.ny)
#
#     if marker is None:
#         # Show marker if only one time point is available,
#         #  otherwise nothing will be shown
#         marker = "o" if len(rdata.t) == 1 else None
#
#     if model is None and rdata.ptr.observable_ids is None:
#         labels = [f"$y_{{{iy}}}$" for iy in observable_indices]
#     elif model is not None and prefer_names:
#         labels = np.asarray(model.getObservableNames())[
#             list(observable_indices)
#         ]
#         labels = [
#             l if l else model.getObservableIds()[ix]
#             for ix, l in enumerate(labels)
#         ]
#     elif model is not None:
#         labels = np.asarray(model.getObservableIds())[list(observable_indices)]
#     else:
#         labels = np.asarray(rdata.ptr.observable_ids)[list(observable_indices)]
#
#     for iy, label in zip(observable_indices, labels, strict=True):
#         (l,) = ax.plot(
#             rdata["t"], rdata["y"][:, iy], marker=marker, label=label
#         )
#
#         if edata is not None:
#             ax.plot(
#                 edata.ts,
#                 edata.observedData[:, iy],
#                 "x",
#                 label=f"exp. {label}",
#                 color=l.get_color(),
#             )
#             ax.errorbar(
#                 edata.ts,
#                 edata.observedData[:, iy],
#                 yerr=rdata.sigmay[:, iy],
#                 fmt="none",
#                 color=l.get_color(),
#             )
#
#     ax.set_xlabel("$t$")
#     ax.set_ylabel("$y(t)$")
#     ax.set_title("Observable trajectories")
#     ax.legend()