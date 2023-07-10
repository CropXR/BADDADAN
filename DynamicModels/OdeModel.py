from __future__ import annotations

import logging
import random
import networkx as nx
import numpy as np
from lmfit import Parameters
from matplotlib import pyplot as plt
from scipy.integrate._ivp.ivp import OdeResult, solve_ivp

from DynamicModels.ModuleRegulatoryNetwork import (ModuleRegulatoryNetwork,
                                                   EdgeRelation)
from DynamicModels.Formula import LinearFormula, NonLinearFormula


class OdeModel:
    """Stores the ODEs for all modules. Can also be used to calculate next time steps."""

    def __init__(self,
                 formula_per_module: list[LinearFormula | NonLinearFormula],
                 is_nonlinear: bool):
        self.formula_per_module = formula_per_module
        self.is_nonlinear = is_nonlinear

    def __repr__(self):
        return ('OdeModel:\n'
                + '\n'.join([f'{formula}'
                             for formula in self.formula_per_module]))

    @property
    def nr_params(self):
        return sum(formula.nr_params for formula in self.formula_per_module)

    @classmethod
    def construct_from_regulatory_network(
            cls, gene_network: ModuleRegulatoryNetwork, nonlinear: bool = False
    ) -> OdeModel:
        """Create ODE network from ModuleRegulatoryNetwork.

        Ensure that it has been converted to a module-module network
        (with .get_module_module_network()).
        """
        formulas = []
        graph = gene_network.graph
        regulatory_directions = [(regulator, module, origin)
                                 for (regulator, module, origin)
                                 in graph.edges(data='origin')]
        if nonlinear:
            assert all(origin in {EdgeRelation.DOWNREGULATES,
                                  EdgeRelation.UP_OR_DOWN,
                                  EdgeRelation.UPREGULATES}
                       for _, _, origin in regulatory_directions), \
                ('Make sure you have removed all TFs from regulatory '
                 'network and converted it to Module-Module network')
        # Iterate over modules in lexicographic order
        for module in sorted(list(graph)):
            if nonlinear:
                regulator_edges = graph.in_edges(module, data='origin')
                formula = NonLinearFormula(module, regulator_edges)
            else:
                regulators = list(graph.predecessors(module))
                formula = LinearFormula(module, regulators)
            formulas.append(formula)
        return cls(formulas, nonlinear)

    def compute_one_step(self, t: float, y: list[float],
                         params: dict[str, float]) -> list[float]:
        """Allows system of ODEs to be called. In this case returns dy/dt
        for all y. Params should be a list which matches the parameter names
        """
        logging.debug(f'Mapped params in the following way: {params}')
        return [formula(t, y, params) for formula in self.formula_per_module]

    def get_module_names(self):
        """Get the names of all modules in the model"""
        return [formula.module_name for formula in self.formula_per_module]

    def get_param_names(self):
        """Get names of all parameters"""
        all_params = []
        for formula in self.formula_per_module:
            all_params.extend(formula.params)
        # Add names of parameters that regulate external temperature
        all_params.extend(['heat_temp', 'non_heat_temp', 'heat_end_time'])
        return all_params

    def add_regulator_to_module(self, target_module_idx: int,
                                origin_module_idx: int = None) -> str | bool:
        """Add a connection to target_module. If origin_module is not
        specified, select a random module.

        :param target_module_idx: Index to use for the target module
        of the connection (e.g. 0 is the first module)
        :param origin_module_idx: If specified, connection
        will be from this module to the target module. If no module is
        specified, pick a random module.
        :return: If connection was added, return the name of the parameter
        that describes the interaction. If no new connection could
        be added, return False.
        """
        # Get candidate regulators first
        candidate_regulators = self.get_module_names()
        if not origin_module_idx:
            # Pick a random regulator
            # Comment this line to assume that module can regulate itself
            candidate_regulators.pop(target_module_idx)
            # Cannot pick modules which are already a regulator
            for regulator in self.formula_per_module[
                target_module_idx].regulator_names:
                candidate_regulators.remove(regulator)
            if candidate_regulators:
                regulator_to_add = random.choice(candidate_regulators)
            else:
                # No modules left to add as regulator
                logging.info('Could not do thickening, module too thicc')
                return False
        else:
            regulator_to_add = candidate_regulators[origin_module_idx]
        new_param_name = self.formula_per_module[
            target_module_idx].add_regulator(regulator_to_add)
        return new_param_name

    def remove_regulator_from_module(self, target_module_idx: int,
                                     origin_module_idx: int = None) -> None:
        """From the module, remove a regulator. If no origin module is provided,
        pick a random one.

        :param target_module_idx: Index of module from which regulator
        should be removed
        :param origin_module_idx: (Optional) Index of module which should be
        removed as regulator of target_module.
        If not provided, a random regulator is removed.
        """
        formula_of_interest = self.formula_per_module[target_module_idx]
        # Get possible regulators
        regulators = formula_of_interest.regulator_names
        assert regulators, (f'<{formula_of_interest}> does not have '
                            f'any regulators. So none can be removed')
        if not origin_module_idx:
            # Randomly pick one to remove
            regulator_to_remove = random.choice(regulators)
        else:
            regulator_to_remove = self.get_module_names()[origin_module_idx]
        formula_of_interest.remove_regulator(regulator_to_remove)

    def flip_regulatory_sign(self, target_module_idx: int,
                             origin_module_idx: int) -> None:
        """Change connection between two modules from inhibition
        to activation or the other way around.
        """
        formula_of_interest = self.formula_per_module[target_module_idx]
        # Only nonlinear formulas can be flipped
        assert isinstance(formula_of_interest, NonLinearFormula)
        origin_module_name = self.get_module_names()[origin_module_idx]
        formula_of_interest.flip_regulatory_direction(origin_module_name)

    def calculate_solution(self, params: Parameters, t: np.ndarray,
                           init_condition_names: list[str]) -> OdeResult:
        """Return values at time points t, given a set of params.
        I.e. this calculates the full solution over time of the system of ODEs.

        :param params: Parameters to use when solving the system of ODEs
        :param t: time points at which to save the solution
        :param init_condition_names: List of strings that are the names of
        the initial conditions. E.g ['y0', 'y1']. These are needed to
        split off from the parameters that are provided
        the solution
        :return: Result of the ODE in OdeResult object
        """
        params = params.valuesdict()
        # Split off initial conditions
        y0 = []
        for name in init_condition_names:
            y0.append(params.pop(name))
        # Split off temp related parameters
        heat_temp = params.pop('heat_temp')
        non_heat_temp = params.pop('non_heat_temp')
        heat_end_time = params.pop('heat_end_time')

        # Solve up until point of heat stress
        t_start = min(t)
        params['temp'] = heat_temp
        t_heat_index = t <= heat_end_time
        t_heat = t[t_heat_index]
        if len(t_heat) > 0:
            # Some time points belong to heat stress
            y_pred_heat = solve_ivp(self.compute_one_step, (t_start, heat_end_time),
                                    y0,
                                    t_eval=t_heat,
                                    args=[params], method='Radau')
            assert y_pred_heat.success, (
                f"Integration failed: {y_pred_heat.message}"
                f"\nParams: {params}")
            # Get new initial conditions
            end_of_heat_expressions = y_pred_heat.y[:, -1].tolist()
        else:
            # No time points belong to heat stress
            end_of_heat_expressions = y0

        # Solve post-heat stress
        t_start = heat_end_time
        params['temp'] = non_heat_temp
        t_non_heat_index = (t > heat_end_time)
        t_non_heat = t[t_non_heat_index]
        y_pred_non_heat = solve_ivp(self.compute_one_step,
                                    (t_start, max(t_non_heat)),
                                    end_of_heat_expressions,
                                    t_eval=t_non_heat,
                                    args=[params], method='Radau')
        assert y_pred_non_heat.success, (
            f"Integration failed: {y_pred_non_heat.message}"
            f"\nParams: {params}")

        if len(t_heat) > 0:
            all_predicted_time = np.concatenate((y_pred_heat.t, y_pred_non_heat.t))
            assert np.all(all_predicted_time == t)
            all_predicted_y = np.concatenate((y_pred_heat.y, y_pred_non_heat.y),
                                             axis=1)
            y_pred_heat.t = all_predicted_time
            y_pred_heat.y = all_predicted_y
            return y_pred_heat
        else:
            # Heat was never applied
            return y_pred_non_heat
