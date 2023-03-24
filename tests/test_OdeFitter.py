import logging

import networkx as nx
import numpy as np
from lmfit import Parameters, fit_report

from DynamicModels.ModuleRegulatoryNetwork import ModuleRegulatoryNetwork, \
    EdgeRelation
from DynamicModels.OdeFitter import OdeFitter
from DynamicModels.OdeModel import OdeModel
from helpers import plot_y_and_y_hat


def test_artificial_data_fit():
    """Generate artificial data from nonlinear model, and fit a new model to it.
    """
    my_graph = nx.DiGraph()
    module_names = [ModuleRegulatoryNetwork.module_prefix + str(i)
                    for i in range(1, 4)]
    my_graph.add_nodes_from(module_names)
    my_graph.add_edge(module_names[0], module_names[1],
                      origin=EdgeRelation.UPREGULATES)
    my_graph.add_edge(module_names[1], module_names[2],
                      origin=EdgeRelation.DOWNREGULATES)
    my_graph.add_edge(module_names[0], module_names[2],
                      origin=EdgeRelation.UPREGULATES)

    articial_network = ModuleRegulatoryNetwork(my_graph)

    articial_network.plot_network()
    artificial_model = OdeModel.construct_from_regulatory_network(
        articial_network,
        nonlinear=True)
    init_condition_names = []
    ground_truth_params = Parameters()
    for param in artificial_model.get_param_names():
        ground_truth_params.add(name=param, value=1)

    ground_truth_params['heat_temp'].set(value=.7)
    ground_truth_params['non_heat_temp'].set(value=.3)
    ground_truth_params['heat_end_time'].set(value=10)

    # Add initial conditions
    for module_name in module_names:
        value = 5 if '1' in module_name else 0
        init_name = f'y{int(module_name[-1])-1}'
        ground_truth_params.add(name=init_name, value=value)
        init_condition_names.append(init_name)
    sim_data = artificial_model.calculate_solution(ground_truth_params,
                                                   t=np.linspace(0, 24),
                                                   init_condition_names=init_condition_names)

    # plot_y_and_y_hat(sim_data.y, sim_data.t)

    my_fitter = OdeFitter(artificial_model, sim_data.y, sim_data.t, heat_end_time=10, param_limit=30)
    # my_fitter.params['heat_temp'].set(value=0, vary=False)
    # my_fitter.params['non_heat_temp'].set(value=0, vary=False)
    my_fitter.fit()
    fit_result = my_fitter.calculate_current_best_fit(sim_data.t)
    plot_y_and_y_hat(sim_data.y, sim_data.t, fit_result)
    return
