import copy
import logging
from pathlib import Path
import re

import mlflow
import matplotlib
import seaborn as sns
import pandas as pd
import pypesto
import petab
import yaml
import amici
from matplotlib import pyplot as plt
from pypesto import optimize as optimize, profile as profile
from pypesto.visualize.model_fit import visualize_optimized_model_fit
import pypesto.petab

from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries


def write_petab_files(expr_mat_time: ExpressionMatrixTimeSeries,
                      sbml_path: Path | str,
                      out_path: Path,
                      experimental_setup: str
                      ):
    """Write necessary petab files for either heat or drought experiment.

    Also see petab documentation online for more info:
    https://petab.readthedocs.io/en/latest/tutorial.html#

    :param expr_mat_time: ExpressionMatrix used to save observations
    :param sbml_path: path to SBML model
    :param out_path: path to save all PETAB files
    :param experimental_setup: describe experimental setup
        (either 'drought' or 'heat')
    """

    assert experimental_setup in ['heat', 'drought'], \
        (f'Currently only support experimental setup for drought or heat.'
         f' Not {experimental_setup=}')

    sbml_importer = amici.SbmlImporter(sbml_path,
                                       show_sbml_warnings=True)
    # observables = amici.assignmentRules2observables(
    #     sbml_importer.sbml,
    #     filter_function=lambda variable: variable.getId().startswith(
    #         "observable_")
    # )

    yaml_dict = write_petab_yaml(experimental_setup, out_path, sbml_path)

    create_parameters_tsv(
        out_path / yaml_dict['parameter_file'], sbml_importer)

    cond_df, expr_mat_cond_name_to_simul_cond_name, noise_level \
        = get_condition_info(experimental_setup)

    create_conditions_tsv(
        out_path / yaml_dict['problems'][0]['condition_files'][0],
        expr_mat_time,
        cond_df
    )

    observables = {f'observable_{species.getId()}':
                       {'formula': species.getId()}
                   for species in sbml_importer.sbml.getListOfSpecies()
                   }

    create_measurements_tsv_heat(
        expr_mat_time,
        out_path / yaml_dict['problems'][0]['measurement_files'][0],
        list(observables.keys()),
        expr_mat_cond_name_to_simul_cond_name
    )
    create_observables_tsv(
        observables,
        out_path / yaml_dict['problems'][0]['observable_files'][0],
        noise_level
    )


def write_petab_yaml(experimental_setup, out_path, sbml_path):
    yaml_dict = {
        'format_version': 1,
        'parameter_file': 'parameters.tsv',
        'problems': [
            {
                'condition_files': ['conditions.tsv'],
                'measurement_files': ['measurements.tsv'],
                'observable_files': ['observable.tsv'],
                'sbml_files': ['../' + Path(sbml_path).name]
            }
        ]
    }
    out_path.mkdir(exist_ok=True)
    with (out_path / f'baddadan_{experimental_setup}_petab.yaml').open(
            'w+') as f:
        yaml.dump(yaml_dict, f)
    return yaml_dict


def get_condition_info(experiment_name: str):
    if experiment_name == 'heat':
        # Create conditions.tsv
        col_names = ['conditionId', 'conditionName', 'temp']
        cond_entries = [
            ['temp21', 'no heat applied', 0],
            ['temp32', 'heat applied', 1]
        ]
        noise_level = .1
        # Translate from expression matrix condition names to names for
        # simulations (because they cant start with a number)
        expr_mat_cond_name_to_simul_cond_name = {
            '21': 'temp21',
            '32': 'temp32'
        }
    elif experiment_name == 'drought':
        col_names = ['conditionId', 'conditionName', 'drought']
        cond_entries = [
            ['control', 'no drought applied', 0],
            ['drought', 'drought applied', 1]
        ]
        noise_level = .1
        # No need to translate here (because conditions dont start with number)
        expr_mat_cond_name_to_simul_cond_name = None
    else:
        raise NotImplementedError
    cond_df = pd.DataFrame(data=cond_entries,
                           columns=col_names)
    return cond_df, expr_mat_cond_name_to_simul_cond_name, noise_level


def create_parameters_tsv(out_path: str | Path,
                          sbml_importer: amici.SbmlImporter):
    """Create parameters CSV for petab"""
    records = []
    for parameter in sbml_importer.sbml.parameters:
        name = parameter.id
        if name in ['drought', 'temp', 'u_t']:
            continue
        parameter_scale = 'log10'
        # parameter_scale = 'lin'
        lb = 0.01
        ub = .5
        if name.startswith('delta_'):
            # Delta always has to be negative (and lower bound
            # then becomes upper bound)
            parameter_scale = 'lin'
            lb, ub = -.5, -0.0001
        elif name.startswith('gamma_'):
            parameter_scale = 'lin'
            lb = -.5
        elif name.startswith('k_'):
            # parameter_scale = 'lin'
            lb = .1
            ub = 10
        elif name.startswith('beta_'):
            parameter_scale = 'log10'
            lb = 0.00001
            ub = 1
        estimate = 1
        records.append([name, parameter_scale, lb, ub, estimate])
    df = pd.DataFrame.from_records(records,
                                   columns=['parameterId',
                                            'parameterScale',
                                            'lowerBound',
                                            'upperBound',
                                            'estimate'])
    df.to_csv(out_path, index=False, sep='\t')


def create_measurements_tsv_heat(
        expr_mat: ExpressionMatrixTimeSeries,
        out_path: str | Path,
        observable_names: list[str],
        condition_name_to_simulation_condition_id: dict = None):
    """Create measurements tsv for PETAB.

    Observable names are the quantities that describe module expressions;
    should be names such as 'observable_y_0', 'observable_y_3'.


    :param expr_mat: Expressionmatrix of experiment
    :param out_path: path to save the measurement.tsv
    :param observable_names: names of observables as specified in the
    observables PETAB file, is used to check that all measurements written
    by this function belong to the observables.
    :param condition_name_to_simulation_condition_id: Condition IDs cannot
    start with numbers, so for heat it needs a dict that maps the
    temperatures (21, 32) to a name such as (temp21, temp32). If dict
    is not provided, just use the simulation names that are returned
    by the ExpressionMatrixTime.
    """
    df_list = []
    for condition_name in expr_mat.condition_names:
        expr_mat_copy = copy.deepcopy(expr_mat)
        expr_mat_copy.keep_only_samples_with_string(condition_name)
        time, df = expr_mat_copy.get_clusters_expressions_with_time(0)
        # df = pd.DataFrame(data)
        df.columns = time
        df.index = df.index.map(lambda x: f'observable_y_{x}')
        assert all(df.index.isin(observable_names))
        df = df.melt(
            ignore_index=False,
            var_name='time',
            value_name='measurement').reset_index(names='observableId')
        if condition_name_to_simulation_condition_id is not None:
            df['simulationConditionId'] = (
                condition_name_to_simulation_condition_id)[condition_name]
        else:
            df['simulationConditionId'] = condition_name
            # Convert time to days to prevent ODE from exploding?
            df['time'] = df['time'] / 24
        correct_order = ['observableId', 'simulationConditionId',
                         'measurement', 'time']
        df = df[correct_order]
        df_list.append(df)

    full_df = pd.concat(df_list)
    full_df.to_csv(out_path, index=False, sep='\t')


def create_observables_tsv(observables: dict, out_path: str | Path, noise_level: float):
    """Create observable TSV for PETAB. Defaults to setting noise to .1 now"""
    col_names_obs = ['observableId', 'observableFormula', 'noiseFormula']
    obs_record = []
    for obs_name, obs_dict in observables.items():
        logging.warning(f"Specified noise as {noise_level}")
        obs_record.append([obs_name, obs_dict['formula'], noise_level])
    obs_df = pd.DataFrame.from_records(obs_record, columns=col_names_obs)
    obs_df.to_csv(out_path, sep='\t', index=False)


def create_conditions_tsv(out_path: str | Path,
                          expr_mat_time: ExpressionMatrixTimeSeries,
                          cond_df: pd.DataFrame):
    """Create conditions TSV for petab"""
    _, data = expr_mat_time.get_clusters_expressions_with_time(0)
    initial_values = data.iloc[:, 0]
    for i, initial_value in initial_values.items():
        species_name = f'y_{i}'
        cond_df[species_name] = initial_value
    # Add them to all this
    cond_df.to_csv(out_path, sep='\t', index=False)


def param_optimise_petab_problem(petab_problem: petab.v1.Problem,
                                 out_folder: Path,
                                 n_starts: int):
    """Given a petab problem, perform parameter optimisation"""
    optimizer, problem = prepare_petab_files_for_fitting(out_folder,
                                                         petab_problem)

    engine = pypesto.engine.MultiProcessEngine()
    result = optimize.minimize(
        problem=problem, optimizer=optimizer, n_starts=n_starts, engine=engine,
    )
    logging.info(result.summary(show_hess=False))

    fig_folder = out_folder / 'figures'
    fig_folder.mkdir(exist_ok=True)
    pypesto.visualize.waterfall(result)
    plt.savefig(fig_folder / 'waterfall_plot.png')
    plt.close()
    pypesto.visualize.parameters(result)
    plt.savefig(fig_folder / 'param_visualise.png')
    plt.close()

    plot_pypesto_module_fit(petab_problem, result, problem)
    # logging.info(return_dict)
    # plt.savefig(fig_folder / 'model_fit.png')
    plt.savefig(fig_folder / 'model_fit.svg')
    plt.close()

    pypesto_out_path = out_folder / 'pypesto_results.hdf5'
    pypesto.store.write_result(
        result=result,
        filename=pypesto_out_path,
        problem=True,
        optimize=True,
        profile=False,
        sample=False,
        overwrite=True
    )

    return result

    # result_loaded = pypesto.store.read_result(pypesto_out_path, optimize=True)

    # param_result = profile.parameter_profile(
    #     problem=problem,
    #     result=result,
    #     optimizer=optimizer,
    #     profile_index=[0, 1],
    # )
    # pypesto.visualize.profile_cis(
    #     param_result, confidence_level=0.95, show_bounds=True
    # )
    # plt.show()
    # pypesto.visualize.optimizer_history(result, trace_y="fval")
    # plt.show()


def prepare_petab_files_for_fitting(
        out_folder: Path,
        petab_problem: petab.v1.Problem
):
    # load from petab_files
    model_folder = out_folder / 'amici_models' / 'baddadan'
    importer = pypesto.petab.PetabImporter(petab_problem,
                                           simulator_type="amici",
                                           output_folder=str(model_folder)
                                           )
    factory = importer.create_objective_creator()
    model = factory.create_model(verbose=False)
    model.setAlwaysCheckFinite(True)
    # some model properties
    print("Model parameters:", list(model.getParameterIds()), "\n")
    print("Model const parameters:", list(model.getFixedParameterIds()), "\n")
    print("Model outputs:   ", list(model.getObservableIds()), "\n")
    print("Model states:    ", list(model.getStateIds()), "\n")
    obj = factory.create_objective()
    # obj.amici_solver.getRelativeTolerance()
    # obj.amici_solver.setRelativeTolerance(
    #     obj.amici_solver.getRelativeTolerance() / 10
    # )
    # obj.amici_solver.setAbsoluteTolerance(
    #     obj.amici_solver.getAbsoluteTolerance() / 10
    # )
    obj.amici_solver.setMaxSteps(10 * obj.amici_solver.getMaxSteps())
    obj.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod.adjoint
    )
    # obj.amici_solver.setAlwaysCheckFinite(True)
    problem = importer.create_problem(obj)
    # # Set gradient computation method to adjoint
    # Maybe switch back? Not sure if it's better
    # optimizer = optimize.ScipyOptimizer()
    optimizer = optimize.ScipyOptimizer(method="L-BFGS-B",
                                        options=dict(maxfun=1e10))
    # engine = pypesto.engine.SingleCoreEngine()
    return optimizer, problem


def plot_nicely_from_artifact(out_folder_of_experiment: str,
                              mlflow_result_uri: str,
                              petab_yaml_path: str):
    """

    :param out_folder_of_experiment: Should contain subdirectory called
    amici_models/baddadan where the amici model lives
    :param mlflow_result_uri: uri to hdf5 result file saved by previous run
    :param petab_yaml_path: path to petab yaml config file
    """
    artifact = mlflow.artifacts.download_artifacts(mlflow_result_uri)
    loaded_result = pypesto.store.read_result(artifact)

    petab_problem = petab.v1.Problem.from_yaml(petab_yaml_path)
    out_folder_of_experiment = Path(out_folder_of_experiment)
    _, problem = prepare_petab_files_for_fitting(out_folder_of_experiment,
                                                 petab_problem)
    # To improve:
    # Standard deviations (?) -> quite hard maybe? ->
    # build upon existing functionality I built earlier maybe?

    plot_pypesto_module_fit(petab_problem, loaded_result, problem)


def plot_pypesto_module_fit(petab_problem, optimise_result, problem):
    # Ensure no fancy sns things remain because they break it for some reason
    matplotlib.rc_file_defaults()
    r_dict = visualize_optimized_model_fit(
        petab_problem=petab_problem,
        result=optimise_result,
        pypesto_problem=problem,
        return_dict=True
    )
    # Map plot names to observable names
    for i, (ax_name, ax) in enumerate(r_dict['axes'].items()):
        legend = ax.get_legend()
        # print()
        match = re.search('y_\d+', legend.get_texts()[0].get_text())
        match_text = match.group(0)
        match_text = match_text.replace('y_', 'Module ')
        ax.set_title(match_text)
        if i < len(r_dict['axes'].items()) - 1:
            legend.remove()
        else:
            legend_entries = legend.get_texts()
            for entry in legend_entries:
                text = entry.get_text()
                parts = text.split('observable')
                if 'simulation' in parts[-1]:
                    parts[-1] = 'Model fit'
                else:
                    parts[-1] = 'Experimental data'
                new_text = ''.join(parts)
                entry.set_text(new_text)
    plt.tight_layout()
    # plt.show()
