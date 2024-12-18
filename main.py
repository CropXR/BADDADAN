import copy
import logging
from pathlib import Path

import yaml
import mlflow

from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from data_wrangling import expr_mat_from_heat, expr_mat_from_drought
from experiment_scripts import (module_size_pipeline, drought_data_to_sbml,
                                heat_data_to_sbml, \
                                config_preprocess, drought_from_wgcna, \
                                wgcna_with_similarity_scores, \
                                save_jackknife_files, \
                                ground_truth_vs_jackknife, pypesto_from_sbml, \
                                analyse_go_enrichments_find_enrichment,
                                do_coherence_with_stat_tests,
                                full_pipeline_prototype)
from exploring_questions import get_coherence_random_modules, \
    get_robustness_random_modules
from figure_pipelines import fig2_from_generated_data
from helpers import plot_y_and_y_hat, get_info_from_gse65046, \
    one_gene_list_file_per_cluster
from petab_integration.petab_scripts import plot_nicely_from_artifact

# pd.options.display.width = 0
# GEOparse.logger.set_verbosity('INFO')

# logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

def main():
    # ONLY EDIT THESE LINES
    # name = '09_heat_data_end_to_end'
    # name = '19_heat_pypesto'
    # name = '22_drought_pypesto'
    # name = '23_coherence_with_stat_tests'
    # name = '24_visualise_fit_result_nicely'
    name = '25_everything_including_limma'
    # name = '26_server_output_visualisation'
    # name = '27_fig2'
    experiment_path = Path(f'data/experiments') / name
    mlflow.set_experiment(name)

    ##  This all shouldn't have to be changed ##

    experiment_path.mkdir(exist_ok=True)
    (experiment_path / "log.log").unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(experiment_path / "log.log"),
                                  logging.StreamHandler()])

    match name:
        case '07_module_size_distribution':
            # FUll module size distributions
            module_size_pipeline(experiment_path)
        case "14_drought_from_wgcna":
            _, hyper_params, _ = config_preprocess(
                experiment_path)
            expr_mat_time, module_module = drought_from_wgcna(experiment_path)
        case "15_wgcna_with_similarity_scores":
            wgcna_with_similarity_scores(experiment_path)
        case "18_robustness_with_wgcna_cutting":
            data_params, hyper_params, experiment_params = config_preprocess(
                experiment_path)
            if 'heat' in data_params['in_path']:
                # Get heat expr_mat
                condition_name = 'heat'
                expr_mat_time = expr_mat_from_heat(data_params['in_path'],
                                                   hyper_params['agg_method'],
                                                   hyper_params['do_log2'])
            elif 'drought' in data_params['in_path']:
                # Get drought expr_mat
                condition_name = 'drought'
                expr_mat_time = expr_mat_from_drought(data_params['in_path'],
                                                      hyper_params[
                                                          'agg_method'],
                                                      hyper_params['do_log2'])
            else:
                raise NotImplementedError(
                    "Couldn't find what condition (drought or heat) was used"
                )
            skip = True
            if not skip:
                save_jackknife_files(
                    experiment_path, expr_mat_time, condition_name)

            #####################
            ## RUN R CODE HERE ##
            #####################

            ground_truth_vs_jackknife(experiment_path, expr_mat_time)

        case "19_heat_pypesto":
            # Run heat data e2e first for this to run:
            heat_data_to_sbml(experiment_path)
            pypesto_from_sbml(experiment_path, 'heat')

        case "20_go_terms_deepsplit_values":
            # for treatment_name in ['heat']:
            # for treatment_name in ['drought']:
            for treatment_name in ['heat', 'drought']:
                data_params, hyper_params, experiment_params = config_preprocess(
                    experiment_path / treatment_name)

                def use_for_analysis(filename: str):
                    """Only keep the deepsplit = 1 or 2 values (because
                    that's all what's needed for the analysis).
                    """
                    if any((f'ds{i}' in filename)
                           for i in hyper_params['r_deep_split']):
                        return True
                    else:
                        return False

                skip = True
                if not skip:
                    one_gene_list_file_per_cluster(
                        in_dir=Path(data_params['in_path']),
                        out_dir=Path(data_params['out_path']),
                        use_for_analysis_func=use_for_analysis
                    )

                go_enrich_output_path = (
                        experiment_path / treatment_name
                        / 'go_outputs_exp_evidence_only_background_de_genes'
                )

                ### RUN SNAKEMAKE ###
                # snakemake - s.. /../../../ snakemake_workflows / Snakefile_wgcna_deepsplit_go_terms - r - c5 - k
                analyse_go_enrichments_find_enrichment(
                    go_enrich_output_path,
                    experiment_path / treatment_name / 'figures',
                    )

                with mlflow.start_run(
                        description=experiment_params['description']):
                    mlflow.log_params(data_params)
                    mlflow.log_params(hyper_params)
                    mlflow.set_tags(experiment_params)
                    mlflow.log_artifact(str(experiment_path / treatment_name / 'figures'))

        case "21_score_distributions_of_random_modules":
            for treatment_name in ['drought', 'heat']:
            # for treatment_name in ['heat']:
                data_params, hyper_params, experiment_params = config_preprocess(
                    experiment_path / treatment_name)

                if 'heat' in data_params['in_path']:
                    condition_name = 'heat'
                    expr_mat_time: ExpressionMatrixTimeSeries = expr_mat_from_heat(
                        data_params['in_path'], hyper_params['agg_method'],
                        hyper_params['do_log2'])
                elif 'drought' in data_params['in_path']:
                    condition_name = 'drought'
                    expr_mat_time: ExpressionMatrixTimeSeries = expr_mat_from_drought(
                        data_params['in_path'], hyper_params['agg_method'],
                        hyper_params['do_log2'])
                else:
                    raise NotImplementedError

                skip = True
                if not skip:
                    expr_mat_copy = copy.deepcopy(expr_mat_time)
                    expr_mat_copy.save_random_modules_for_goa_find_enrichment(
                        wgcna_label_file = data_params['wgna_label_file'],
                        out_dir=Path(data_params['split_by_module_out_path'])
                    )

                ### RUN SNAKEMAKE ###
                # snakemake - s.. /../../../ snakemake_workflows / Snakefile_wgcna_deepsplit_go_terms - r - c5 - k

                analyse_go_enrichments_find_enrichment(
                    # Path('data/experiments/21_score_distributions_of_random_modules/drought/go_outputs_exp_evidence_only'),
                    Path(data_params['split_by_module_out_path']).parent / 'go_outputs_exp_evidence_only',
                    experiment_path / treatment_name / 'figures',
                )

                jackknife_paths = (
                    list(Path(
                        data_params['jacknife_path'])
                         .glob(data_params['jacknife_glob_command'])
                         )
                )
                get_robustness_random_modules(
                    jackknife_paths=jackknife_paths,
                    full_dataset_path=data_params['wgna_label_file'],
                    figure_out_dir=Path(data_params['fig_out_path']))

                get_coherence_random_modules(
                    wgcna_label_file=data_params['wgna_label_file'],
                    expr_mat_time=expr_mat_time,
                    figure_out_dir=Path(data_params['fig_out_path'])
                )

                # Get the scores for the local / global / combined scores -> maybe just the combined scores?

                # Do statistical tests to see what they are like / and / or calculate module-specific Z-scores

                with mlflow.start_run(
                        description=experiment_params['description']):
                    mlflow.log_params(data_params)
                    mlflow.log_params(hyper_params)
                    mlflow.set_tags(experiment_params)
                    mlflow.log_artifact(
                        str(experiment_path / treatment_name / 'figures'))

        case '22_drought_pypesto':
            drought_data_to_sbml(experiment_path)
            pypesto_from_sbml(experiment_path, 'drought')

        case '23_coherence_with_stat_tests':
            for treatment_name in ['drought', 'heat']:
                data_params, hyper_params, experiment_params = config_preprocess(
                    experiment_path / treatment_name)

                if 'heat' in data_params['in_path_expr_mat']:
                    expr_mat_time: ExpressionMatrixTimeSeries = expr_mat_from_heat(
                        data_params['in_path_expr_mat'],
                        hyper_params['agg_method'], hyper_params['do_log2'])
                elif 'drought' in data_params['in_path_expr_mat']:
                    expr_mat_time: ExpressionMatrixTimeSeries = expr_mat_from_drought(
                        data_params['in_path_expr_mat'],
                        hyper_params['agg_method'], hyper_params['do_log2'])
                else:
                    raise NotImplementedError
                do_coherence_with_stat_tests(
                    in_dir=Path(data_params['in_path_clusterings']),
                    expr_mat_time=expr_mat_time,
                    out_dir=Path(data_params['out_path'])
                )
                with mlflow.start_run(
                        description=experiment_params['description']):
                    mlflow.log_params(data_params)
                    mlflow.log_params(hyper_params)
                    mlflow.set_tags(experiment_params)
                    mlflow.log_artifact(
                        str(experiment_path / treatment_name / 'figures'))
        case '24_visualise_fit_result_nicely':
            config_path = experiment_path / 'config.yaml'
            with config_path.open('r') as f:
                config_dict = yaml.safe_load(f)
            for name, condition_dict in config_dict.items():
                artifact = mlflow.artifacts.download_artifacts(
                    condition_dict['mlflow_result_uri'])
                plot_nicely_from_artifact(
                    condition_dict['out_folder_experiment'],
                    artifact,
                    condition_dict['petab_yaml']
                    )
        case '25_everything_including_limma':
            full_pipeline_prototype(experiment_path)
        case "26_server_output_visualisation":
            config_path = experiment_path / 'config.yaml'
            with config_path.open('r') as f:
                config_dict = yaml.safe_load(f)
            for name, condition_dict in config_dict.items():
                plot_nicely_from_artifact(
                    condition_dict['out_folder_experiment'],
                    condition_dict['artifact_path'],
                    condition_dict['petab_yaml']
                )
        case '27_fig2':
            fig2_from_generated_data(experiment_path.parent / '25_everything_including_limma')

        case _:
            raise NotImplementedError(f'{name} not found')


if __name__ == "__main__":
    main()
