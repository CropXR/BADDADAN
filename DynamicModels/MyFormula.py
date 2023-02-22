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
        self.regulator_names = regulator_names
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

    def add_regulator(self, regulator):
        regulator_index = int(regulator[-1]) - 1
        b_param_name = f'beta_{regulator_index}_{self.module_index}'
        self.params.append(b_param_name)
        # Currently assume linear relationship
        self.formula_string += f'+ {b_param_name} * y[{regulator_index}]'
        self.formula_is_compiled = False
        return b_param_name
