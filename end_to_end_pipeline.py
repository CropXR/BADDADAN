import logging
from pathlib import Path

import mlflow

from experiment_scripts import save_files_for_wgcna_cutting, \
    do_coherence_with_stat_tests, analyse_go_enrichments_find_enrichment, \
    from_expr_mat_time_to_ode, save_supp_table_go_enrichments, \
    pypesto_from_sbml
from figure_pipelines import see_gene_module_sizes
from expr_mat_factories import expr_mat_time_factory
from helpers import one_gene_list_file_per_cluster, config_preprocess


def full_pipeline_prototype(experiment_path: Path):
    """Main script: do all processing from input data to output model in one go"""
    skip_slow_steps = True
    # for treatment_name in ['heat']:
    # for treatment_name in ['drought']:
    for treatment_name in ['drought', 'heat']:
        logging.info(f'Doing {treatment_name}')
        treatment_path = experiment_path / treatment_name
        data_params, hyper_params, experiment_params = config_preprocess(
            treatment_path)
        # # Uncomment to only save files
        # with mlflow.start_run(
        #         description=experiment_params['description']):
        #     mlflow.log_params(data_params)
        #     mlflow.log_params(hyper_params)
        #     mlflow.set_tags(experiment_params)
        #     mlflow.log_artifact(
        #         str(treatment_path / 'figs'))
        #     mlflow.log_artifact(
        #         str(treatment_path / 'petab_files'))
        #     mlflow.log_artifact(
        #         str(treatment_path / 'pypesto_results.hdf5'))
        #     mlflow.log_artifact(
        #         str(treatment_path / 'log.log'))
        #     # mlflow.log_artifact(
        #     #     str(treatment_path / 'go_terms_supp_table.csv'))
        # continue

        if not skip_slow_steps:
            expr_mat_all_genes = expr_mat_time_factory(
                treatment_path,
                data_params['soft_path'],
                hyper_params['agg_method'],
                hyper_params['do_log2'],
                gpl_path=data_params.get('gpl_path', None)
            )
            expr_mat_all_genes.save_for_limma(treatment_path / '01_input_for_limma.csv')

        # ## Here: run limma script (limma_de_selection/de_selection.R) ##
        # continue
        # Select only the DE genes
        de_file_path = list(treatment_path.glob('02[a_]*.csv'))
        assert len(de_file_path) == 1
        de_file_path = str(de_file_path[0])
        expr_mat_time = expr_mat_time_factory(
            treatment_path,
            de_file_path,
            hyper_params['agg_method'],
            hyper_params['do_log2'],
            gpl_path=None)
        #
        expr_mat_time.merge_biological_samples()
        # # Read DE genes from limma output and get the ATTED/Merged/Local scores
        # g = sns.clustermap(a.iloc[:2000, :2000], cmap='coolwarm', cbar_pos=None,
        #                    dendrogram_ratio=0);
        # g.ax_heatmap.set_xticks([]);
        # g.ax_heatmap.set_yticks([]);
        # plt.tight_layout();
        # # plt.show()

        skip_slow_steps = True
        if not skip_slow_steps:
            save_files_for_wgcna_cutting(treatment_path, data_params, expr_mat_time)
        ## Here: run wgcna cutting script (r_wgcna_dyntreecut/dyntreecut.R) ##
        # continue
            see_gene_module_sizes(expr_mat_time,
                                  cut_modules_path=treatment_path / 'dyntreecut_output',
                                  figure_path=treatment_path / 'figs')


        if not skip_slow_steps:
            one_gene_list_file_per_cluster(
                in_dir=treatment_path / 'dyntreecut_output',
                out_dir=treatment_path / 'split_by_module',
                use_for_analysis_func=lambda x: True
            )
        # Also generate random clusters that have the same size as a representative of these clusters
        if not skip_slow_steps:
            expr_mat_time.save_random_modules_for_goa_find_enrichment(
                wgcna_label_file=treatment_path
                                 / 'dyntreecut_output'
                                 / 'combined_sum_dists_wgcna_clustered_ds1.csv',
                out_dir=treatment_path / 'split_by_module'
            )

        # Coherence
        if not skip_slow_steps:
            do_coherence_with_stat_tests(
                in_dir=treatment_path / 'split_by_module',
                expr_mat_time=expr_mat_time,
                out_dir=treatment_path / 'figs'
            )

        # continue
        # Do GO enrichment
        ### RUN SNAKEMAKE ###
        # snakemake -s ../../../../snakemake_workflows/Snakefile_wgcna_deepsplit_go_terms -r -c5 -k

        go_enrich_output_path = (
                treatment_path
                / 'go_outputs_exp_evidence_only_background_de_genes'
        )
        if not skip_slow_steps:
            analyse_go_enrichments_find_enrichment(
                go_enrich_output_path,
                treatment_path / 'figs',
                )

        sbml_path = treatment_path / 'module_network.xml'
        skip_slow_steps = False
        if not skip_slow_steps:
            # ODE modelling steps
            my_ode = from_expr_mat_time_to_ode(data_params, treatment_path,
                                               expr_mat_time, hyper_params)

            # These are parameters that are different between the two datasets
            u_t_function = 'temp' if treatment_name == 'heat' \
                else 'drought * time / 13'
            my_ode.save_to_sbml(sbml_path,
                                u_t_function)
        skip_slow_steps = True
        if not skip_slow_steps:
            # Save GO enrich files into one file
            expr_mat_pickl_path = treatment_path / 'expr_mat_time.pkl'
            save_supp_table_go_enrichments(expr_mat_pickl_path,
                                           go_enrich_output_path,
                                           treatment_path)
        #
        # # This is done on server now
        # pypesto_from_sbml(treatment_path,
        #                   treatment_name,
        #                   treatment_path / 'expr_mat_time.pkl',
        #                   sbml_path,
        #                   )

                          # use_best_params_as_init= treatment_path / 'pypesto_results.hdf5')

        with mlflow.start_run(
                description=experiment_params['description']):
            mlflow.log_params(data_params)
            mlflow.log_params(hyper_params)
            mlflow.set_tags(experiment_params)
            mlflow.log_artifact(
                str(treatment_path / 'figs'))
