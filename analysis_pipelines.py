import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import dill as pickle
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries, \
    AggregationMethod
from data_wrangling import expr_mat_from_drought, expr_mat_from_emexp, \
    merge_ath_annotation_for_goatools
from helpers import get_info_from_gse65046

@dataclass
class ClusteringArgs:
    """Arguments to use for clustering with different linkage matrices"""
    input_dist_name: str
    linkage_matrix_path: Path | None
    original_dist_path: Path | None


def module_network_from_tf2_output(expr_mat_time,
                                   tf2_in_path,
                                   tf2_out_path,
                                   threshold,
                                   module_plot_path):
    my_grn = ModuleRegulatoryNetwork.from_tf2_tsv(tf2_out_path)
    my_grn.add_tf_module_mappings(tf2_in_path,
                                  from_tf2_input=True)
    my_grn.keep_only_modules_of_interest(expr_mat_time)
    my_grn.clean_up_network()
    my_grn.check_if_tfs_created_by_module(expr_mat_time,
                                          do_plotting=False,
                                          remove_low_corr=False,
                                          assert_correlated=False)
    my_grn.set_up_or_downregulation(expr_mat_time, do_plotting=False,
                                    threshold=threshold)
    # my_grn.plot_network(nx.d  raw_kamada_kawai, with_labels=False)
    module_module = my_grn.get_module_module_network()
    # # module_module.graph = nx.create_empty_copy(module_module.graph, with_data=False)
    module_module.plot_network(nx.draw_kamada_kawai , with_labels=True, out_path=module_plot_path)
    logging.info(list(module_module.graph.edges(data=True)))
    return module_module
