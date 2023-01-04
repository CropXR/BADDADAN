import logging

import networkx as nx
import numpy as np


class OdeInference:
    """Stores the ODEs for all modules. Can also be used to calculate next time steps."""
    def __init__(self):
        self.nr_params = 0
        self.formula_per_module: list[MyFormula] = []

    def __repr__(self):
        # TODO needs some work perhaps
        return '\n'.join([f'{formula}' for formula in self.formula_per_module])

    def __call__(self, t: float, y: list[float], *params: list[float]):
        """Allows system of ODEs to be called. In this case returns dy/dt
        for all y. Params should be a list which matches the parameter names"""
        output = []
        assert len(params) == self.nr_params, 'Supplied nr of parameters does not match actual number of parameters'
        param_dict = dict(zip(self.get_param_names(), params))
        # logging.info(f'Mapped params in the following way: {param_dict}')
        for formula in self.formula_per_module:
            # TODO map this in a cooler, more efficient way. Maybe speeds up calculations, too.
            output.append(formula(y, param_dict))
        return output

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


class MyFormula:
    """Formula for a single module, can be provided with custom parameter values when called."""
    def __init__(self, module_name: str, regulator_names: list[str], init_i: int):
        self.module = module_name
        self.module_index = int(module_name[-1]) - 1
        self.params = []
        formula_segments = []
        self.next_param_suffix = init_i
        for regulator in regulator_names:
            regulator_index = int(regulator[-1]) - 1
            b_param_name = f'beta{self.next_param_suffix}'
            k_param_name = f'k{self.next_param_suffix}'
            self.params.extend([b_param_name, k_param_name])
            # TODO does this currently also work correctly for inhibition? Not sure...
            formula_segments.append(f'({b_param_name} * y[{regulator_index}]) '
                                    f'/ ({k_param_name} + y[{regulator_index}])')
            self.next_param_suffix += 1
        # Only add terms if >1 regulator
        self.formula_string = ' + '.join(formula_segments)
        # Add decay factor
        d_param_name = f'd{init_i}'
        self.params.append(d_param_name)
        self.formula_string += f' - {d_param_name} * y[{self.module_index}]'
        # Compile string to speed up evaluation
        self.compiled_formula_string = compile(self.formula_string,
                                               "<string>", "eval")

    @property
    def nr_params(self):
        """Get the number of parameters"""
        return len(self.params)

    def __repr__(self):
        return f'Formula of {self.module}={self.formula_string} ' \
               f'\n nr_params = {self.nr_params}'

    def __call__(self, y: list[float], params: dict[str, float]) -> float:
        """When called, calculate the outcome of the formula"""
        # Create dict for all these things
        init_val = {"y": y}
        local_dict = init_val | params
        return eval(self.compiled_formula_string, {}, local_dict)
