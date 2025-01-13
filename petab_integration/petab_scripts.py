import copy
import logging
from pathlib import Path
import re

import dill as pickle
import mlflow
import matplotlib
import numpy as np
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
from scipy.interpolate import CubicSpline, UnivariateSpline

from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries


def write_petab_files(expr_mat_time: ExpressionMatrixTimeSeries,
                      sbml_path: Path | str,
                      out_path: Path,
                      experimental_setup: str,
                      do_interpolate: bool = False,
                      param_guess_dict: dict | None = None,
                      do_extra_datapoints: bool = False
                      ):
    """Write necessary petab files for either heat or drought experiment.

    Also see petab documentation online for more info:
    https://petab.readthedocs.io/en/latest/tutorial.html#

    :param expr_mat_time: ExpressionMatrix used to save observations
    :param sbml_path: path to SBML model
    :param out_path: path to save all PETAB files
    :param experimental_setup: describe experimental setup
        (either 'drought' or 'heat')
    :param do_interpolate: If true, interpolate the data using a spline
    """

    assert experimental_setup in ['heat', 'drought'], \
        (f'Currently only support experimental setup for drought or heat.'
         f' Not {experimental_setup=}')

    sbml_importer = amici.SbmlImporter(sbml_path,
                                       show_sbml_warnings=False)
    # observables = amici.assignmentRules2observables(
    #     sbml_importer.sbml,
    #     filter_function=lambda variable: variable.getId().startswith(
    #         "observable_")
    # )
    cond_df, expr_mat_cond_name_to_simul_cond_name, noise_level \
        = get_condition_info(experimental_setup)

    yaml_dict = write_petab_yaml(experimental_setup, out_path, sbml_path,
                                 do_interpolate)

    create_parameters_tsv(
        out_path / yaml_dict['parameter_file'], sbml_importer,
        parameter_guess_dict=param_guess_dict
    )

    observables = {f'observable_{species.getId()}':
                       {'formula': species.getId()}
                   for species in sbml_importer.sbml.getListOfSpecies()
                   }
    measurement_file_path = out_path / yaml_dict['problems'][0]['measurement_files'][0]
    create_measurements_tsv_heat(
        expr_mat_time,
        measurement_file_path,
        list(observables.keys()),
        expr_mat_cond_name_to_simul_cond_name,
        do_interpolate=do_interpolate,
        do_extra_datapoints=do_extra_datapoints
    )

    # When interpolated, the initial values are not necessarily the same,
    # so no need to check that
    create_conditions_tsv(
        out_path / yaml_dict['problems'][0]['condition_files'][0],
        measurement_file_path,
        cond_df,
        assert_same_initial_values = not do_interpolate
    )

    create_observables_tsv(
        observables,
        out_path / yaml_dict['problems'][0]['observable_files'][0],
        noise_level
    )


def write_petab_yaml(experimental_setup, out_path, sbml_path,
                     do_interpolate: bool = False):
    measurement_file_name = 'measurements.tsv' if not do_interpolate \
        else 'interpolated_measurements.tsv'

    yaml_dict = {
        'format_version': 1,
        'parameter_file': 'parameters.tsv',
        'problems': [
            {
                'condition_files': ['conditions.tsv'],
                'measurement_files': [measurement_file_name],
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
                          sbml_importer: amici.SbmlImporter,
                          parameter_guess_dict: dict[str:float] = None):
    """Create parameters CSV for petab"""
    records = []
    for parameter in sbml_importer.sbml.parameters:
        name = parameter.id
        if name in ['drought', 'temp', 'u_t']:
            continue
        if name.startswith('delta_'):
            parameter_scale = 'lin'
            lb, ub = -50, 0
        elif name.startswith('gamma_'):
            parameter_scale = 'lin'
            lb, ub = -10, 10
        elif name.startswith('k_'):
            parameter_scale = 'log10'
            lb, ub = 0.1, 10
        elif name.startswith('beta_'):
            parameter_scale = 'log10'
            lb, ub = 0.000001, 1000
        elif name.startswith('a_'):
            parameter_scale = 'log10'
            lb, ub = 0.000001, 100
        elif name.startswith('phi_'):
            parameter_scale = 'lin'
            lb, ub = -np.pi, np.pi
        elif name.startswith('b_'):
            parameter_scale = 'lin'
            lb, ub = -10, 10
        else:
            raise NotImplementedError('Does not know what param limits to set')
        do_estimate = 1
        records.append([name, parameter_scale, lb, ub, do_estimate])
    df = pd.DataFrame.from_records(records,
                                   columns=['parameterId',
                                            'parameterScale',
                                            'lowerBound',
                                            'upperBound',
                                            'estimate'])
    if parameter_guess_dict:
        df['initializationPriorType'] = 'parameterScaleNormal'
        def mapping_func(param_id):
            if param_id in parameter_guess_dict:
                param_guess = parameter_guess_dict[param_id]
                standard_dev = .5
            else:
                param_guess = 1
                standard_dev = 5
            # standard_dev = .1 if in_row['parameterScale'] == 'lin' else -1
            return f"{param_guess};{standard_dev}"
        df['initializationPriorParameters'] = df['parameterId'].map(
            mapping_func)
    df.to_csv(out_path, index=False, sep='\t')


def create_measurements_tsv_heat(
        expr_mat: ExpressionMatrixTimeSeries,
        out_path: str | Path,
        observable_names: list[str],
        condition_name_to_simulation_condition_id: dict = None,
        do_interpolate: bool = False,
        do_extra_datapoints: bool = False):
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
    :param do_interpolate: Interpolate the data
    """
    expressions = expr_mat.extract_module_expressions_long_form()
    expression_list = expr_mat.split_series_into_different_conditions(
        expressions)

    df = pd.concat(expression_list)

    df = df.rename(columns={'cluster_id': 'observableId',
                            'expression': 'measurement',
                            'condition': 'simulationConditionId'})
    df = df.drop('elapsed_mins', axis=1)
    df['observableId'] = df['observableId'].map(lambda x: f'observable_y_{x}')
    assert all(df['observableId'].isin(observable_names))

    if condition_name_to_simulation_condition_id is not None:
        # This is the heat data
        # Convert time to hours
        df['time'] = df['time'].dt.total_seconds() / (60 * 60)
        df['simulationConditionId'] = df['simulationConditionId'].map(
            condition_name_to_simulation_condition_id)
    else:
        # This is the drought data
        assert df['simulationConditionId'].isin(['drought']).any()
        # Convert time to days to prevent ODE from exploding?
        df['time'] = df['time'].dt.days
    correct_order = ['observableId', 'simulationConditionId',
                     'measurement', 'time']
    df = df[correct_order]

    if do_interpolate and do_extra_datapoints:
        raise ValueError("Can't have both do_interpolate "
                         "and do_extra_datapoints set to true")
    elif do_interpolate:
        groups = df.groupby(['observableId', 'simulationConditionId'])
        record_list = []
        for (obs_id, sim_cond), group in groups:
            time = group['time'].values
            measurement = group['measurement'].values
            # 5 degrees of freedom
            spline = UnivariateSpline(time, measurement,
                                      s=5)
            # interpolate for each time points
            time_fine = np.linspace(time.min(), time.max(), 23)
            measurement_fine = spline(time_fine)
            record_list.extend(
                [[obs_id, sim_cond, y, t]
                 for (y, t) in zip(measurement_fine, time_fine)]
            )
            # # Plot the data and the fitted spline
            # plt.figure(figsize=(6, 4))
            # plt.plot(time, measurement, 'o',
            #          label=f'{obs_id} - {sim_cond} (data)')
            # plt.plot(time_fine, measurement_fine, 'x',
            #          label=f'{obs_id} - {sim_cond} (spline)')
            # plt.legend()
            # plt.xlabel('Time')
            # plt.ylabel('Measurement')
            # plt.title(f'Spline Fit for {obs_id} - {sim_cond}')
            # plt.show()


        df_interpolation = pd.DataFrame.from_records(record_list)  # -> save this to measurements as well
        df_interpolation.columns = df.columns
        df = df_interpolation
        # df_interpolation.to_csv(
        #     out_path.with_name('interpolation_measurement.tsv'),
        #     index=False, sep='\t')
    elif do_extra_datapoints:
        # Get two latest datapoints
        timepoints = df['time'].unique()
        timepoints.sort()
        new_row_list = []
        for late_timepoint in timepoints[-2:]:
            #Add copies of time point (ie more weight) to last two time points
            # to balance out uneven sampling
            for _, row in df[df['time'] == late_timepoint].iterrows():
                for _ in range(4):
                    new_row_list.append(row)
        df = pd.concat([df, pd.DataFrame(new_row_list)],
                       ignore_index=True)

    df.to_csv(out_path, index=False, sep='\t')


def create_observables_tsv(observables: dict, out_path: str | Path, noise_level: float):
    """Create observable TSV for PETAB."""
    col_names_obs = ['observableId', 'observableFormula', 'noiseFormula']
    obs_record = []
    for obs_name, obs_dict in observables.items():
        logging.warning(f"Specified noise as {noise_level}")
        obs_record.append([obs_name, obs_dict['formula'], noise_level])
    obs_df = pd.DataFrame.from_records(obs_record, columns=col_names_obs)
    obs_df.to_csv(out_path, sep='\t', index=False)


def create_conditions_tsv(out_path: str | Path,
                          measurement_file_path: Path,
                          cond_df: pd.DataFrame,
                          assert_same_initial_values = False):
    """Create conditions TSV for petab"""
    df = pd.read_csv(measurement_file_path, sep='\t')
    initial_values_per_cluster = df[df['time'] == 0]
    initial_values_per_cluster = initial_values_per_cluster.groupby(
        ['observableId', 'time'])
    cond_df = cond_df.set_index('conditionId')
    if assert_same_initial_values:
        assert all(initial_values_per_cluster['measurement'].std() < 1e-9)
    for index, init_val_one_cluster_df in initial_values_per_cluster:
        species_name = re.findall('y_\d+', index[0])[0]
        init_val_one_cluster_df = init_val_one_cluster_df[
            ['simulationConditionId', 'measurement']]
        for _, one_row in init_val_one_cluster_df.iterrows():
            cond_id = one_row['simulationConditionId']
            measurement = one_row['measurement']
            cond_df.loc[cond_id, species_name] = measurement
    cond_df.to_csv(out_path, sep='\t')


def param_optimise_petab_problem(petab_problem: petab.v1.Problem,
                                 out_folder: Path,
                                 n_starts: int):
    """Given a petab problem, perform parameter optimisation"""
    optimizer, problem = prepare_petab_files_for_fitting(out_folder,
                                                         petab_problem)

    engine = pypesto.engine.MultiProcessEngine()
    # engine = pypesto.engine.SingleCoreEngine()

    result = optimize.minimize(
        problem=problem, optimizer=optimizer, n_starts=n_starts, engine=engine,
    )
    logging.info(result.summary(show_hess=False))

    fig_folder = out_folder / 'figs'
    fig_folder.mkdir(exist_ok=True)
    pypesto.visualize.waterfall(result)
    plt.savefig(fig_folder / 'waterfall_plot.svg')
    plt.close()
    pypesto.visualize.parameters(result)
    plt.savefig(fig_folder / 'param_visualise.png')
    plt.close()

    plot_pypesto_module_fit(petab_problem, result, problem)

    petab_problem_og = petab.v1.Problem.from_yaml(
        'data/experiments/25_everything_including_limma/heat/petab_files/baddadan_heat_petab.yaml'
    )

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
                              artifact,
                              petab_yaml_path: str,
                              expr_mat_time_path = None):
    """
    :param out_folder_of_experiment: Should contain subdirectory called
    amici_models/baddadan where the amici model lives
    :param artifact: hdf5 result file saved by previous run
    :param petab_yaml_path: path to petab yaml config file
    """
    loaded_result = pypesto.store.read_result(artifact)

    petab_problem = petab.v1.Problem.from_yaml(petab_yaml_path)
    out_folder_of_experiment = Path(out_folder_of_experiment)
    _, problem = prepare_petab_files_for_fitting(out_folder_of_experiment,
                                                 petab_problem)
    # To improve:
    # Standard deviations (?) -> quite hard maybe? ->
    # build upon existing functionality I built earlier maybe?
    if expr_mat_time_path:
        with expr_mat_time_path.open('rb') as f:
            expr_mat_time: ExpressionMatrixTimeSeries = pickle.load(f)
        expr_mat_time.add_constant(3, do_all_pos_check=False)
        # get_std_per_cluster
        ci_per_cluster = expr_mat_time.get_ci_per_cluster(confidence_level=.99, for_error_bars=True)
        split_ci_list = expr_mat_time.split_series_into_different_conditions(ci_per_cluster.T)
        # std_per_cluster = expr_mat_time.get_std_per_cluster()
        # split_std_list = expr_mat_time.split_series_into_different_conditions(std_per_cluster.T)
    plot_pypesto_module_fit(
        petab_problem, loaded_result, problem, error_bar_size_list=split_ci_list)


def plot_pypesto_module_fit(petab_problem, optimise_result, problem, error_bar_size_list = None):
    # Ensure no fancy sns things remain because they break it for some reason
    matplotlib.rc_file_defaults()
    r_dict = visualize_optimized_model_fit(
        petab_problem=petab_problem,
        result=optimise_result,
        pypesto_problem=problem,
        return_dict=True
    )
    if 'drought' in petab_problem.condition_df.index:
        condition_name_dict = {'control_keyword': 'control',
         'treatment_keyword': 'drought'}
    else:
        condition_name_dict = {'control_keyword': '21',
         'treatment_keyword': '32'}
    # Map plot names to observable names
    for i, (ax_name, ax) in enumerate(r_dict['axes'].items()):
        legend = ax.get_legend()
        # print()
        match = re.search('y_\d+', legend.get_texts()[0].get_text())
        match_text = match.group(0)
        match_text = match_text.replace('y_', 'Module ')
        ax.set_title(match_text, fontsize=32)
        module_id = match_text.split(' ')[1]
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=16)  # X-axis ticks
        ax.tick_params(axis='y', labelsize=16)

        # Error bars
        if error_bar_size_list is not None:
            for line in ax.get_lines():
                if line.get_marker() == 'x':
                    # Experimental data
                    line_color = line.get_color()
                    errors_for_condition = None
                    for error_bars in error_bar_size_list:
                        if any(error_bars['condition'] == condition_name_dict['control_keyword']) and line_color == '#1f77b4': # Blue is control condition
                            errors_for_condition = error_bars
                            break
                        elif any(error_bars['condition'] == condition_name_dict['treatment_keyword']) and line_color == '#ff7f0e': # Orange is heat condition
                            errors_for_condition = error_bars
                            break
                        # else:
                        #     raise NotImplementedError("Could not find which errorbars to use")
                    x, y = line.get_xdata(), line.get_ydata()
                    assert errors_for_condition is not None, 'Could not find which errorbars to use'
                    errors = errors_for_condition[int(module_id)]
                    if len(errors) == len(y) - 1:
                        # Missing first time point (only in control)
                        y = y[1:]
                        x = x[1:]
                    if errors.shape[1] == 2:
                        # Upper and lower limits
                        errors = errors.T
                    ax.errorbar(x, y,
                                yerr=errors,
                                color=line_color, fmt=' ')
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
