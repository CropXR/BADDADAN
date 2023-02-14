import logging
import re
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.optimize import minimize as scipy_minimize
from lmfit import minimize as lmfit_minimize

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from helpers import de_print_fun


def plot_net(graph):
    node_color_map = ['blue' if ('TF' in node) else 'orange'
                      for node in graph.nodes]

    edge_mapping_dict = {'regulates': 'green',
                         'binds_to': 'red',
                         'transcribed_by': 'black',
                         'Activates': 'green',
                         'Represses': 'red'}

    edge_color_map = [edge_mapping_dict[origin] for _, _, origin
                      in graph.edges.data('origin')]

    nx.draw_networkx(graph,
                     font_size=6,
                     font_weight='bold',
                     font_color='magenta',
                     node_color=node_color_map,
                     edge_color=edge_color_map,
                     pos=nx.kamada_kawai_layout(graph))
    plt.show()


def ODE_generator(a_graph):
    '''
    Transcribes to linear diff. equation.

    There is one ODE per module.

    It takes the graph with *just the modules* and usses the predecessor
    modules for each module to generate a function. Every ODE has a decay variable also.

    It returns variable and parameter names besides the functional relationships.
    '''
    module_formulas = {}
    parameters = []
    variables = []
    constants = []

    def exct_m_num(module_string):
        '''
        Extracts module number
        '''
        return re.findall(r'\d+', module_string)[0]

    for module in sorted(list(a_graph)):
        formula_segments = []

        regulators = list(a_graph.predecessors(module))

        # Add regulatory relationships:
        for regulator in regulators:
            parameter = f'beta_{exct_m_num(regulator)}_{exct_m_num(module)}'
            variable = f'y_{exct_m_num(regulator)}'

            if parameter not in parameters:
                parameters.append(parameter)
            if variable not in variables:
                variables.append(variable)

            regulator_expression = f'+ {parameter} * {variable}'
            formula_segments.append(regulator_expression)

        # Add decay:
        decay_parameter = f'decay_{exct_m_num(module)}'
        decay_variable = f'y_{exct_m_num(module)}'

        formula_segments.append(f'- {decay_parameter} * {decay_variable}')

        # Add production based on temperature
        temp_parameter = f'gamma_{exct_m_num(module)}'
        temp_variable = f'temp'

        formula_segments.append(f'+ {temp_parameter} * {temp_variable}')

        if decay_parameter not in parameters:
            parameters.append(decay_parameter)

        if decay_variable not in variables:
            variables.append(decay_variable)

        if temp_parameter not in parameters:
            parameters.append(temp_parameter)

        if temp_variable not in constants:
            constants.append(temp_variable)

        module_formulas[f'dy_{exct_m_num(module)}/dt ='] = ' '.join(
            formula_segments)

    for formula in module_formulas:
        logging.info(f'{formula} {module_formulas[formula]}')

    parameters.sort()
    variables.sort()
    constants.sort()
    return module_formulas, parameters, variables, constants


def calculate_ODEs_outcome(t, y0, variable_names, params_dict, formulas,
                           constants):
    '''
    Returns the value for the modules differential expression
    given a set of parameters

    To use eval it is needed to provide a dictionary with the arguments necesary to evaluate
    the string. Here the variable values (e.g. initial conditions) and parameter values are
    used to evaluate every formula.
    '''

    def time_to_heatstress(t: float):

        if t < 3:  # Hours
            temp = 1.
        else:
            temp = .69
        return [temp]

    constants_dict = dict(zip(constants, time_to_heatstress(t)))

    # assign initial conditions to be evaluated
    variable_dict = dict \
        (zip(variable_names, y0))

    local_dict = variable_dict | params_dict | constants_dict  # Merge dictionaries to evaluate

    return [eval(formulas[formula], {}, local_dict) for formula in formulas]


def num_solve_ODE(outcomes_function, y0, time_span, params_dict, formulas,
                  variable_names, constants, t_eval=None):
    """Solve the ODE numerically given parameter values

    Simply use a fancy euler's method to calculate y given t:
    i.e. y = y0 + epsilon*dy kind of thing
    (notice that in the linear settings analytic solution is easily computable!)
    """
    return solve_ivp(fun=outcomes_function,
                     t_span=time_span,
                     y0=y0,
                     t_eval=t_eval,
                     args=(variable_names, params_dict, formulas, constants))


def graph_preprocessing(in_path: Path) -> nx.DiGraph:
    assert in_path.is_absolute(), 'Make sure the path is absolute'
    aracne_net = pd.read_csv(in_path / 'aracne_network_edges.csv')
    aracne_net['origin'] = 'binds_to'
    my_graph = nx.from_pandas_edgelist(aracne_net, source='regulator',
                                       target='target',
                                       edge_attr='origin',
                                       create_using=nx.DiGraph)

    # plot_net(my_graph)

    # We know which TF determine which MODULES but we have to add the origins!
    original_connections = nx.read_edgelist(
        in_path / 'my_clustering_edgelist.csv',
        create_using=nx.DiGraph)

    original_connections = original_connections.reverse()
    my_graph.add_edges_from(original_connections.edges,
                            origin='transcribed_by')
    # plot_net(my_graph)

    # Now, this graph has to be cleaned:
    # This means that me must remove bidirectional edges (self-regulation, which is impossible to tell with this approach ??),
    # TF that are expressed but do not bind to any module (This can happen because we have removed the 'binds_to' edge of bidirectional TF)
    # and remove nodes that do not transcribe or bind anything.

    # To use the cleaning functionality
    my_graph_class = ModuleRegulatoryNetwork(my_graph)
    my_graph_class.clean_up_network()
    my_graph_class = my_graph_class.get_module_module_network()
    return my_graph_class.graph


def calculate_MSE(params, param_names, variable_names, formulas,
                  constants, y, time_points):
    param_dict = dict(
        zip(param_names, params))  # Notice this is not random anymore!

    solution = num_solve_ODE(outcomes_function=calculate_ODEs_outcome,
                             y0=y[:, 0],
                             time_span=[time_points[0], time_points[-1]],
                             t_eval=time_points,
                             variable_names=variable_names,
                             params_dict=param_dict,
                             formulas=formulas,
                             constants=constants)

    assert solution.success, f"The solver broke down: {solution.message}"
    y_hat = solution.y

    # y = average_module_expression.to_numpy()
    mse = np.mean((y - y_hat) ** 2)
    logging.debug(f'{mse=}')

    return mse


def solve_ODE(y, time_points, in_graph, loadsatime=False):
    formulas, parameters, variables, constants = ODE_generator(in_graph)
    # Currently hardcoded
    bounds = [  # Betas
        (-10, 10), (-10, 10),
        (-10, 10), (-10, 10),
        (-10, 10),
        # In order for this to make sense the order of the parameters is KEY!
        # Decay
        (0, 10), (0, 10),
        (0, 10), (0, 10),
        # Temp growth
        (-10, 10), (-10, 10),
        (-10, 10), (-10, 10), ]
    assert len(bounds) == len(
        parameters), "Assign one upper and lower bound for each parameter"

    optimize_result = differential_evolution(func=calculate_MSE,
                                             bounds=bounds,
                                             args=(
                                                 parameters, variables,
                                                 formulas,
                                                 constants, y, time_points),
                                             # x0=np.random.randn(len(parameters)),
                                             callback=de_print_fun,
                                             polish=False,
                                             popsize=8,
                                             workers=8,
                                             maxiter=10)
    # optimize_result = lmfit_minimize(calculate_MSE,
    #                                  params=...,
    #
    #                                  kws={'t': self.time_points,
    #                                       'y_real': self.measured_data,
    #                                       't_start': t_start,
    #                                       't_end': t_end},
    #                                  args=(parameters, variables, formulas,
    #                                        constants, y, time_points),
    #                                  method='differential_evolution',
    #                                  fit_kws={"callback": de_print_fun,
    #                                           "polish": False,
    #                                           "popsize": 8,
    #                                           "workers": 8,
    #                                           "maxiter": 10}
    #                                  )
    # optimize_result = scipy_minimize(fun=calculate_MSE,
    #                            x0=np.random.randn(len(parameters)),
    #                            args=(parameters, variables, formulas,
    #                                  constants, y, time_points),
    #                            bounds=bounds,
    #                            method='L-BFGS-B'
    #                            )
    # assert optimize_result.success, f'Optimisation failed: {optimize_result.message}'
    logging.info('========================')
    logging.info(optimize_result)
    logging.info('========================')
    logging.info('========================')

    param_dict = dict(zip(parameters, optimize_result.x))
    og_time = time_points
    if loadsatime:
        time_points = np.linspace(0, 24, 50)
    ode_result = num_solve_ODE(outcomes_function=calculate_ODEs_outcome,
                               y0=y[:, 0],
                               time_span=[time_points[0], time_points[-1]],
                               t_eval=time_points,
                               variable_names=variables,
                               params_dict=param_dict,
                               formulas=formulas,
                               constants=constants)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(15, 8)
    fig.suptitle('Comparison real vs estimated')
    for row in ode_result.y:
        ax1.plot(ode_result.t, row)
        ax1.set_title('y_hat')

    for row in y:
        ax2.plot(og_time, row)
        ax2.set_title('y')

    plt.show()
    # plt.close()

    return optimize_result, ode_result


def retrying_from_scratch(in_path: Path,
                          measured_expressions: ExpressionMatrixTimeSeries):
    my_graph = graph_preprocessing(in_path)

    # # Uncomment to try with fully connected graph
    # my_graph = nx.complete_graph(4, create_using=nx.DiGraph)
    # mapping_dict = {i: f'MODULE{i}' for i in my_graph.nodes}
    # my_graph = nx.relabel_nodes(my_graph, mapping_dict)

    n_clusters = 4
    measured_expressions.keep_only_shoot()
    measured_expressions.merge_biological_samples()
    measured_expressions.keep_only_de_genes(std_cutoff=1.5)
    col_names, average_module_expression = \
        measured_expressions.get_clusters_expressions_with_time(n_clusters)

    optim_result, ode_result = solve_ODE(average_module_expression, col_names,
                                         my_graph, loadsatime=True)

    second_fit_result, fitting_to_sim_values = solve_ODE(ode_result.y,
                                                         ode_result.t, my_graph)
    logging.info('Original params')
    logging.info(optim_result.x)
    logging.info('Guessed params')
    logging.info(second_fit_result.x)
