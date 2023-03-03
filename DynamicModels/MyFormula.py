import logging


class MyFormula:
    """
    Formula for a single module, can be provided with custom parameter values
    when called.
    """
    def __init__(self, module_name: str, regulator_names: list[str]):
        """
        :param module_name: Name of module that formula describes
        :param regulator_names: List of all modules that regulate this module
        """
        self.module_name = module_name
        self.module_index = int(module_name[-1]) - 1
        self.params = []
        self.regulator_names = []
        self.formula_string = ''
        for regulator in regulator_names:
            self.add_regulator(regulator)

        # Add decay factor, its suffix is always equal to the module it belongs to
        d_param_name = f'delta_{self.module_index}'
        self.params.append(d_param_name)
        self.formula_string += f' - {d_param_name} * y[{self.module_index}]'

        # Add factor which expresses the change in expression based on temperature
        gamma_param_name = f'gamma_{self.module_index}'
        self.params.append(gamma_param_name)
        self.formula_string += f' + {gamma_param_name} * temp'

        # Add names of parameters that regulate relation between heat and non-heat temperature
        self.params.extend(['heat_temp', 'non_heat_temp'])
        # Register if the module has been compiled (which speeds up its evaluation)
        self.formula_is_compiled = False
        self.compile_formula()

    def compile_formula(self):
        # Compile string to speed up evaluation
        self.compiled_formula_string = compile(self.formula_string.lstrip(),
                                               "<string>", "eval")
        self.formula_is_compiled = True

    @property
    def nr_params(self):
        """Get the number of parameters"""
        return len(self.params)

    @staticmethod
    def generate_linear_term(param_name: str, var_name: str,
                             is_positive: bool = True) -> str:
        """For a given parameter name and variable name, return the mathematical
        expresssion that linearly describes their relationship.

        :param param_name: Name of the parameter which describes the relationship
        :param var_name: Name of variable that should be multiplied
        with the parameter
        :param is_positive: If true, the expression will start with a '+'. If
        false the expression will start with a '-'.
        :return: example: the string '+ beta_2_0 * y[0]'
        """
        operator = '+' if is_positive else '-'
        return f'{operator} {param_name} * {var_name}'

    def __repr__(self):
        return f'Formula of {self.module_name}={self.formula_string} ' \
               f'\n nr_params = {self.nr_params}'

    def __call__(self, t: float, y: list[float], params: dict[str, float]) -> float:
        """When called, calculate the outcome of the formula

        :param t: Current timepoint
        :param y: List of expressions of all modules
        :param params: Dict that maps parameter names to their value
        :return: Derivative at time point t,
                  given parameters and expressions of modules.
        """
        # Create dict for all these things
        temperature = {"temp": self.time_to_heatstress(t, params)}
        init_val = {"y": y}
        # Merge all dicts
        local_dict = init_val | params | temperature
        if not self.formula_is_compiled:
            self.compile_formula()
        result = eval(self.compiled_formula_string, {}, local_dict)
        return result

    def time_to_heatstress(self, t: float, params: dict) -> float:
        # Assume that t is in hours.
        if t < 3:
            logging.debug(f'Still in heatstress 🔥🔥🔥🔥 {t=}')
            temp = params['heat_temp']
        else:
            logging.debug(f'No longer in heatstress ❄❄❄❄ {t=}')
            temp = params['non_heat_temp']
        return temp

    def add_regulator(self, regulator: str):
        """Add a regulator (e.g. MODULE2) to the equation. And returns name of
        the new parameter

        :param regulator: Name of regulator module, e.g. MODULE3
        :return: Name of the new parameter
        """
        regulator_index = int(regulator[-1]) - 1
        b_param_name = f'beta_{regulator_index}_{self.module_index}'
        self.regulator_names.append(regulator)
        self.params.append(b_param_name)
        # Currently assume linear relationship
        self.formula_string += self.generate_linear_term(
            b_param_name, f'y[{regulator_index}]')
        self.formula_is_compiled = False
        return b_param_name

    def remove_regulator(self, regulator_to_remove: str):
        """Remove the regulator from the equation. Only works if regulator
        is already present in the formula.

        :param regulator_to_remove: Name of regulator module, e.g. MODULE3
        :return: Name of the new parameter"""
        regulator_index = int(regulator_to_remove[-1]) - 1
        b_param_name = f'beta_{regulator_index}_{self.module_index}'
        assert regulator_to_remove in self.regulator_names,\
            'Regulator cannot be removed, it is not present in the current formula'
        self.params.remove(b_param_name)
        self.regulator_names.remove(regulator_to_remove)
        string_to_remove_from_formula = self.generate_linear_term(
            b_param_name, f'y[{regulator_index}]')
        self.formula_string = self.formula_string.replace(
            string_to_remove_from_formula, '')
        self.formula_is_compiled = False
        return True
