from typing import Tuple, Generator, Callable

import numpy as np
from lmfit import Parameters
from networkx.classes.reportviews import InEdgeDataView
from scipy.integrate._ivp.ivp import OdeResult

from DynamicModels.ModuleRegulatoryNetwork import EdgeRelation
from Formula.FormulaSuperClass import FormulaSuperClass

class NonLinearFormula(FormulaSuperClass):
    def __init__(self, module_name: str,
                 regulator_edges: InEdgeDataView):
        super().__init__(module_name)
        for regulator, _, direction in regulator_edges:
            self.add_regulator(regulator, direction)
        self.compile_formula()

    def get_hill_equation_outcomes(
            self, current_fit: OdeResult, params: Parameters,
            t: np.ndarray) -> Generator[Tuple[str, float, float, float], None, None]:
        """For a current fit, see if the full dynamics of the hill equations
        are covered.

        :param current_fit: Result of fitting ODE to experimental data
        :param params: Parameters for which to investigate the
        :param t:
        :return: Generator which yields tuples containing:
                    - String which describes the interaction that the
                    hill equation represents (e.g. MODULE1->MODULE2)
                    - Time point at which the hill equation is evaluated
                    - The absolute outcome of the hill equation
                    - The normalized outcome of the hill equation (i.e.
                    divided by beta, so the value is between 0 and 1)
        """
        regulator_count = 0
        for formula_part in self.formula_parts:
            if not formula_part.lstrip().startswith('+ (beta'):
                # Not a hill equation
                continue
            for i, time_point in enumerate(t):
                all_params = params.valuesdict() | {'y': current_fit.y[:, i]}
                outcome = eval(formula_part, {}, all_params)
                raise NotImplementedError('Might not be able to handle'
                                          ' module names with two digit indices')
                norm_outcome = outcome / all_params[f'beta_{self.regulator_names[regulator_count][-1]}_{self.module_name[-1]}']
                yield (f'{self.regulator_names[regulator_count]}->{self.module_name}',
                       time_point, outcome, norm_outcome)
            regulator_count += 1

    def add_regulator(self, regulator: str, direction: EdgeRelation):
        """Add a regulator (e.g. MODULE2) to the equation. And returns name of
        the new parameter

        :param regulator: Name of regulator module, e.g. MODULE3
        :return: Name of the new parameter
        """
        b_param_name, k_param_name, var_name = self.generate_param_and_var_names(
            regulator)
        self.regulator_names.append(regulator)
        self.params.extend([b_param_name, k_param_name])
        if direction == EdgeRelation.UPREGULATES:
            term_to_add = self.generate_hill_activation_term(
                b_param_name, k_param_name, var_name, n=2)
        elif direction == EdgeRelation.DOWNREGULATES:
            term_to_add = self.generate_hill_inhibition_term(
                b_param_name, k_param_name, var_name, n=2)
        else:
            raise NotImplementedError('If regulatory interaction is unclear, cannot use that properly yet')
        self.formula_parts.append(term_to_add)
        self.formula_is_compiled = False
        return b_param_name

    def generate_param_and_var_names(self, regulator: str) -> Tuple[str, str, str]:
        """For a given regulator, generate what the names of beta, k,
         and the input variable (e.g. y[1]) should be.
         """
        # Assume module index is integer at end of module name
        regulator_index = self.module_name_to_index(regulator)
        b_param_name = f'beta_{regulator_index}_{self.module_index}'
        k_param_name = f'k_{regulator_index}_{self.module_index}'
        var_name = f'y[{regulator_index}]'
        return b_param_name, k_param_name, var_name

    def remove_regulator(self, regulator_to_remove: str):
        """Remove the regulator from the equation. Only works if regulator
        is already present in the formula.

        :param regulator_to_remove: Name of regulator module, e.g. MODULE3
        :return: Name of the new parameter"""
        b_param_name, k_param_name, var_name = self.generate_param_and_var_names(
            regulator_to_remove)
        assert regulator_to_remove in self.regulator_names, \
            'Regulator cannot be removed, it is not present in the current formula'
        self.params.remove(b_param_name)
        self.params.remove(k_param_name)
        self.regulator_names.remove(regulator_to_remove)
        activation_string_to_remove = self.generate_hill_activation_term(
            b_param_name, k_param_name, var_name)
        inhibition_string_to_remove = self.generate_hill_inhibition_term(
            b_param_name, k_param_name, var_name)
        if activation_string_to_remove in self.formula_parts:
            self.formula_parts.remove(activation_string_to_remove)
        elif inhibition_string_to_remove in self.formula_parts:
            self.formula_parts.remove(inhibition_string_to_remove)
        else:
            raise KeyError('Trying to remove term from '
                           'formula that does not exist')

        self.formula_is_compiled = False
        return True

    def flip_regulatory_direction(self, origin_module_name: str):
        """Change an activation into inhibition or vice versa.

        Used to check how sensitive the model is to inhibition/activation
        assumptions."""
        b_param_name, k_param_name, var_name = self.generate_param_and_var_names(
            origin_module_name)
        assert origin_module_name in self.regulator_names, \
            'Regulator cannot be removed, it is not present in the current formula'
        activation_string = self.generate_hill_activation_term(
            b_param_name, k_param_name, var_name)
        inhibition_string = self.generate_hill_inhibition_term(
            b_param_name, k_param_name, var_name)
        if activation_string in self.formula_parts:
            self.formula_parts.remove(activation_string)
            self.formula_parts.append(inhibition_string)
        elif inhibition_string in self.formula_parts:
            self.formula_parts.remove(inhibition_string)
            self.formula_parts.append(activation_string)
        else:
            raise KeyError('Trying to remove term from '
                           'formula that does not exist')
        self.formula_is_compiled = False

