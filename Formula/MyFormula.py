import logging


class MyFormula:
    """
    Formula for a single module, can be provided with custom parameter values when called.
    """
    def __init__(self, module_name: str, regulator_names: list[str], init_i: int):
        """
        :param module_name: Name of module that formula describes
        :param regulator_names: List of all modules that regulate this module
        :param init_i: Initial suffix for parameters to use
        """
        self.module_name = module_name
        self.module_index = int(module_name[-1]) - 1
        self.params = []
        formula_segments = []
        self.next_param_suffix = init_i
        for regulator in regulator_names:
            # TODO create some .add_term() method or something?
            regulator_index = int(regulator[-1]) - 1
            b_param_name = f'beta{self.next_param_suffix}'
            self.params.append(b_param_name)
            # Currently assume linear relationship
            # if self.next_param_suffix == 3:
            formula_segments.append(
                f'{b_param_name} * y[{regulator_index}] * temp')
            # else:
            #     formula_segments.append(
            #         f'{b_param_name} * y[{regulator_index}]')
            self.next_param_suffix += 1
        # Only add terms if >1 regulator
        self.formula_string = ' + '.join(formula_segments)

        # Add decay factor, its suffix is always equal to the module it belongs to
        d_param_name = f'd{self.module_index}'
        self.params.append(d_param_name)
        self.formula_string += f' - {d_param_name} * y[{self.module_index}]'

        # # Add factor which expresses the change in expression based on temperature
        # t_param_name = f't{self.module_index}'
        # self.params.append(t_param_name)
        # self.formula_string += f' + {t_param_name} * temp'

        # Compile string to speed up evaluation
        self.compiled_formula_string = compile(self.formula_string,
                                               "<string>", "eval")

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
        temperature = {"temp": self.time_to_heatstress(t)}
        init_val = {"y": y}
        # Merge all dicts
        local_dict = init_val | params | temperature
        result = eval(self.compiled_formula_string, {}, local_dict)
        return result

    def time_to_heatstress(self, t: float) -> float:
        # Assume that t is in minutes. And temperature was 30 degrees during the first 3 hours, 22 degrees after that.
        # TODO pull these values from the self.parameters object?
        if t < 3*60:
            logging.debug(f'Still in heatstress 🔥🔥🔥🔥 {t=}')
            temp = 1.
        else:
            logging.debug(f'No longer in heatstress ❄❄❄❄ {t=}')
            temp = .7
        return temp
