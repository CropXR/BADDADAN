import logging

import networkx as nx
import numpy as np


class OdeInference:
    """Stores the ODEs for all modules. Can also be used to calculate next time steps."""
    def __init__(self):
        self.nr_params = 0
        self.formula_per_module: list[MyFormula] = []

    def construct_formula_per_module(self, graph: nx.DiGraph):
        """For each module, generate a formula based on the connectivity of the module in the graph"""
        for module in graph.nodes:
            regulators = list(graph.predecessors(module))
            formula = MyFormula(module, regulators, init_i=self.nr_params // 2)
            self.formula_per_module.append(formula)
            self.nr_params += formula.nr_params

    def get_param_names(self):
        """Get names of all parameters"""
        all_params = []
        for formula in self.formula_per_module:
            all_params.extend(formula.params)
        return all_params

    def __call__(self, t, y: list[float], *params: list[float]):
        """Allows system of ODEs to be called. In this case returns dy/dt
        for all y. Params should be a list which matches the parameter names"""
        output = []
        assert len(*params) == self.nr_params, 'Supplied nr of parameters does not match actual number of parameters'
        param_dict = dict(zip(self.get_param_names(), *params))
        logging.info(f'Mapped params in the following way: {param_dict}')
        for formula in self.formula_per_module:
            output.append(formula(y, param_dict))
        return output


class MyFormula:
    """Formula for a single module, can be provided with custom parameter values."""
    def __init__(self, module_name: str, regulator_names: list[str], init_i: int):
        self.module = module_name
        self.params = []
        formula_segments = []
        for regulator in regulator_names:
            regulator_index = int(regulator[-1]) - 1
            b_param_name = f'beta{init_i}'
            k_param_name = f'k{init_i}'
            self.params.extend([b_param_name, k_param_name])
            formula_segments.append(f'({b_param_name} * y[{regulator_index}]) '
                                    f'/ ({k_param_name} + y[{regulator_index}])')
            init_i += 1
        # Only add terms if >1 regulator
        self.formula_string = ' + '.join(formula_segments)
        # Compile string to speed up evaluation
        self.compiled_formula_string = compile(self.formula_string,
                                               "<string>", "eval")

    @property
    def nr_params(self):
        """Get the number of parameters"""
        return len(self.params)

    def __repr__(self):
        return f'Formula of {self.module}={self.formula_string} \n nr_params = {self.nr_params}'

    def __call__(self, y: list[float], params: dict[str, float]) -> float:
        """When called, calculate the outcome of the formula"""
        # Create dict for all these things
        init_val = {"y": y}
        local_dict = init_val | params
        return eval(self.compiled_formula_string, {}, local_dict)
