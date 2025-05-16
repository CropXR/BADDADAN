import logging
from pathlib import Path

import yaml
import mlflow
from matplotlib import pyplot as plt

from experiment_scripts import sa_drought_over_time, \
    compare_parameters_between_hill_orders
from end_to_end_pipeline import full_pipeline_prototype
from figure_pipelines import fig2_from_generated_data, \
    run_pypesto_model_with_different_params
from petab_integration.petab_scripts import plot_nicely_from_artifact

# pd.options.display.width = 0
# GEOparse.logger.set_verbosity('INFO')

# logging.basicConfig(level=logging.DEBUG)
# Local mlflow server for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

def main():
    # ONLY EDIT THESE LINES
    # name = "24_visualise_fit_result_nicely"
    # name = '25_everything_including_limma'
    # name = '26_server_output_visualisation'
    # name = '27_fig2'
    # name = '28_sa_in_drought'
    # name = '29_different_params'
    name = '30_response_to_reviewers'
    experiment_path = Path(f'data/experiments') / name
    mlflow.set_experiment(name)

    ##  This all shouldn't have to be changed ##

    experiment_path.mkdir(exist_ok=True)
    (experiment_path / "log.log").unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(experiment_path / "log.log"),
                                  logging.StreamHandler()])

    match name:
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
                    condition_dict['petab_yaml'],
                    Path(condition_dict['expr_mat_pkl'])
                    )
                plt.savefig(condition_dict['fig_out_path'])
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
        case '28_sa_in_drought':
            sa_drought_over_time()
        case '29_different_params':
            config_path = experiment_path / 'config.yaml'
            with config_path.open('r') as f:
                config_dict = yaml.safe_load(f)
            for name, condition_dict in config_dict.items():
                artifact = mlflow.artifacts.download_artifacts(
                    condition_dict['mlflow_result_uri'])
                run_pypesto_model_with_different_params(
                    condition_dict['out_folder_experiment'],
                    artifact,
                    condition_dict['petab_yaml'],
                    condition_dict['param_names'],
                    condition_dict['new_value']
                    )
            with mlflow.start_run():
                mlflow.log_artifact(str(experiment_path))
        case '30_response_to_reviewers':
            # with mlflow.start_run():
            #     mlflow.log_artifact(str(experiment_path / 'hill_order_1'))
            #     mlflow.log_artifact(str(experiment_path / 'hill_order_3'))
            # compare_parameters_between_hill_orders()
            full_pipeline_prototype(experiment_path)

        case _:
            raise NotImplementedError(f'{name} not found')


if __name__ == "__main__":
    main()
