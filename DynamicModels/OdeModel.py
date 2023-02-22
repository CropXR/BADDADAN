import logging
import random

import networkx as nx

from DynamicModels.MyFormula import MyFormula


class OdeModel:
    """Stores the ODEs for all modules. Can also be used to calculate next time steps."""
    def __init__(self):
        self.nr_params = 0
        self.formula_per_module: list[MyFormula] = []

    def __repr__(self):
        return ('OdeModel:\n'
                + '\n'.join([f'{formula}' for formula in self.formula_per_module]))

    def __call__(self, t: float, y: list[float],
                 params: dict[str, float]) -> list[float]:
        """Allows system of ODEs to be called. In this case returns dy/dt
        for all y. Params should be a list which matches the parameter names
        """
        logging.debug(f'Mapped params in the following way: {params}')
        return [formula(t, y, params) for formula in self.formula_per_module]

    def get_module_names(self):
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

    def add_random_regulator_to_module(self, module_idx: int):
        # Get candidate regulators first
        candidate_regulators = self.get_module_names()
        # # Uncomment to assume that module cannot regulate itself
        # candidate_regulators.pop(module_idx)
        # Cannot pick modules which are already a regulator
        for regulator in self.formula_per_module[module_idx].regulator_names:
            candidate_regulators.remove(regulator)
        if candidate_regulators:
            regulator_to_add = random.choice(candidate_regulators)
            new_param_name = self.formula_per_module[module_idx].add_regulator(regulator_to_add)
            return new_param_name
        else:
            logging.info('Could not do thickening, module too thicc')
            return False
