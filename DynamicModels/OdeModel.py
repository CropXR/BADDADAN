import logging
import random

import networkx as nx
import numpy as np
from lmfit import Parameters
from scipy.integrate._ivp.ivp import OdeResult, solve_ivp

from DynamicModels.MyFormula import MyFormula


class OdeModel:
    """Stores the ODEs for all modules. Can also be used to calculate next time steps."""
    def __init__(self):
        self.nr_params = 0
        self.formula_per_module: list[MyFormula] = []

    def __repr__(self):
        return ('OdeModel:\n'
                + '\n'.join([f'{formula}' for formula in self.formula_per_module]))

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

    def construct_formula_per_module(self, graph: nx.DiGraph):
        """
        For each module, generate a formula based on the connectivity of
        the module in the graph
        """
        # Iterate over modules in lexicographic order
        for module in sorted(list(graph)):
            regulators = list(graph.predecessors(module))
            formula = MyFormula(module, regulators)
            self.formula_per_module.append(formula)
            self.nr_params += formula.nr_params

    def get_param_names(self):
        """Get names of all parameters"""
        all_params = []
        for formula in self.formula_per_module:
            all_params.extend(formula.params)
        return all_params

    def add_regulator_to_module(self, target_module_idx: int,
                                origin_module_idx: int = None) -> str | bool:
        """Add a connection to target_module. If origin_module is not
        specified, select a random module.

        :param target_module_idx: Index to use for the target module
        of the connection (e.g. 0 is the first module)
        :param origin_module_idx: If specified, connection
        will be from this module to the target module.
        :return: If connection was added, return the name of the parameter
        that describes the interaction. If no new connection could
        be added, return False.
        """
        # Get candidate regulators first
        candidate_regulators = self.get_module_names()
        if not origin_module_idx:
            # Pick a random regulator
            # Uncomment this line to assume that module cannot regulate itself
            candidate_regulators.pop(target_module_idx)
            # Cannot pick modules which are already a regulator
            for regulator in self.formula_per_module[target_module_idx].regulator_names:
                candidate_regulators.remove(regulator)
            if candidate_regulators:
                regulator_to_add = random.choice(candidate_regulators)
            else:
                # No modules left to add as regulator
                logging.info('Could not do thickening, module too thicc')
                return False
        else:
            regulator_to_add = candidate_regulators[origin_module_idx]
        new_param_name = self.formula_per_module[target_module_idx].add_regulator(regulator_to_add)
        return new_param_name

    def calculate_solution(self, params: Parameters,
                           t: np.ndarray, init_condition_names: list[str],
                           t_start=None,
                           t_end=None) -> OdeResult:
        """Return values at time points t, given a set of params

        :param params: Parameters to use when solving the system of ODEs
        :param t: time points at which to save the solution
        :param init_condition_names: List of strings that are the names of
        the initial conditions. E.g ['y0', 'y1']
        :param t_start: (Optional), time point to start calculation of
        the solution
        :param t_end: (Optional), time point to end calculation of the solution
        :return: Result of the ODE in OdeResult object
        """
        if t_start is None:
            t_start = min(t)
        if t_end is None:
            t_end = max(t)
        if t_start is not None or t_end is not None:
            selected_time_points = (t_start <= t) & (t <= t_end)
            t = t[selected_time_points]
        # Split off initial conditions
        y0 = [value for name, value in params.valuesdict().items()
              if name in init_condition_names]
        param_dict = {
            name: value for name, value in params.valuesdict().items()
            if name not in init_condition_names
        }
        y_pred = solve_ivp(self.compute_one_step, (t_start, t_end), y0,
                           t_eval=t,
                           args=[param_dict], method='Radau')
        assert y_pred.success, (f"Integration failed: {y_pred.message}"
                                f"\nParams: {params}")
        return y_pred
