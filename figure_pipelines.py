import copy
import re
from pathlib import Path

import mlflow
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from Expressions.ExpressionMatrix import AggregationMethod, \
    ExpressionMatrixTimeSeries
from experiment_scripts import do_coherence_with_stat_tests, \
    analyse_go_enrichments_find_enrichment, \
    plot_gene_modules_ds_size_distribution, plot_module_size_distributions
from expr_mat_factories import expr_mat_time_factory


def fig2_from_generated_data(experiment_path):
    sns.set_theme()
    fig, axes = plt.subplots(2, 2,
                             sharex='col', sharey='row', figsize=(7, 5.5))
    for ax_index, treatment_name in enumerate(['drought', 'heat']):
        treatment_path = experiment_path / treatment_name
        de_file_path = list(treatment_path.glob('02[a_]*.csv'))
        assert len(de_file_path) == 1
        de_file_path = str(de_file_path[0])
        expr_mat_time = expr_mat_time_factory(
            treatment_path,
            de_file_path,
            AggregationMethod.MEAN,
            False,
            gpl_path=None)
        #
        expr_mat_time.merge_biological_samples()
        do_coherence_with_stat_tests(
            in_dir=treatment_path / 'split_by_module',
            expr_mat_time=expr_mat_time,
            out_dir=None,
            ax_to_plot_on=axes[0][ax_index]
        )
        go_enrich_output_path = (
                treatment_path
                / 'go_outputs_exp_evidence_only_background_de_genes'
        )
        analyse_go_enrichments_find_enrichment(
            in_path=go_enrich_output_path,
            out_path=None,
            ax_to_plot_on=axes[1][ax_index]
        )
    # plt.tight_layout()
    for ax in axes.flat:
        ax.set_ylim(0, 1)
        ax.set_xlabel('')

    for ax in axes[:, 1]:
        ax.set_ylabel('')

    tick_label_map = {'atted_dists': 'Global',
                      'combined_sum_dists': 'Combined',
                      'local_dists': 'Local',
                      'random': 'Random'}
    for ax in axes[1, :]:
        # Get the current x-axis tick labels
        current_labels = ax.get_xticklabels()
        # Modify the tick labels based on the dictionary
        new_labels = [tick_label_map.get(label.get_text(), label.get_text())
                      for label in current_labels]
        ax.set_xticklabels(new_labels)

    plt.savefig(experiment_path / 'fig2.svg',    bbox_inches = 'tight')


def see_gene_module_sizes(expr_mat_time: ExpressionMatrixTimeSeries,
                          cut_modules_path: Path,
                          figure_path: Path):
    out_records = []
    for dyntreecut_file in cut_modules_path.iterdir():
        expr_mat_time_copy = copy.deepcopy(expr_mat_time)
        expr_mat_time_copy.assign_clusters_from_wgcna(dyntreecut_file)
        sizes = expr_mat_time_copy.get_module_sizes()
        method, ds_value = dyntreecut_file.name.split('_wgcna_clustered_')
        ds_value = re.search('(?<=ds)\d+', ds_value).group()
        for size in sizes:
            out_records.append((size, method, ds_value))
    df = pd.DataFrame.from_records(out_records,
                                   columns=['module_size', 'method',
                                            'deepsplit'])
    plot_gene_modules_ds_size_distribution(df, figure_path)


def module_size_pipeline(experiment_path):
    for file in experiment_path.iterdir():
        if file.name.endswith('expr_mat_dict.pkl'):
            plot_module_size_distributions(file)
    with mlflow.start_run():
        for file in experiment_path.iterdir():
            mlflow.log_artifact(str(file))
            # if not file.suffix in ['.npy', '.pkl', '.gzip']:
            #     mlflow.log_artifact(str(file))
