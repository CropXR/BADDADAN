import logging

import networkx as nx

from MyFormula import MyFormula


class OdeModel:
    """Stores the ODEs for all modules. Can also be used to calculate next time steps."""
    def __init__(self):
        self.nr_params = 0
        self.formula_per_module: list[MyFormula] = []

    def __repr__(self):
        return ('OdeInferenceClass:\n'
                + '\n'.join([f'{formula}' for formula in self.formula_per_module]))

    def __call__(self, t: float, y: list[float], *params: float):
        """Allows system of ODEs to be called. In this case returns dy/dt
        for all y. Params should be a list which matches the parameter names"""
        assert len(params) == self.nr_params, 'Supplied nr of parameters does not match actual number of parameters'
        param_dict = dict(zip(self.get_param_names(), params))
        logging.debug(f'Mapped params in the following way: {param_dict}')
        return [formula(y, param_dict) for formula in self.formula_per_module]

    def construct_formula_per_module(self, graph: nx.DiGraph):
        """For each module, generate a formula based on the connectivity of the module in the graph"""
        # Iterate over modules in lexicographic order
        for module in sorted(list(graph)):
            regulators = list(graph.predecessors(module))
            if len(self.formula_per_module) == 0:
                init_i = 0
            else:
                init_i = self.formula_per_module[-1].next_param_suffix
            formula = MyFormula(module, regulators, init_i=init_i)
            self.formula_per_module.append(formula)
            self.nr_params += formula.nr_params

    def get_param_names(self):
        """Get names of all parameters"""
        all_params = []
        for formula in self.formula_per_module:
            all_params.extend(formula.params)
        return all_params
