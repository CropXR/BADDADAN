from matplotlib import pyplot as plt
import seaborn as sns

from Expressions.ExpressionMatrix import AggregationMethod
from experiment_scripts import expr_mat_time_factory, \
    do_coherence_with_stat_tests, analyse_go_enrichments_find_enrichment


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
