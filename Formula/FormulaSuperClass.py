class FormulaSuperClass:
    """Class can only be inherited, not used by itself"""

    def __init__(self, module_name: str):
        """
        :param module_name: Name of module that formula describes
        :param regulator_names: List of all modules that regulate this module
        """
        self.module_name = module_name
        self.module_index = int(module_name[-1])
        self.params = []
        self.regulator_names = []
        # Capture each term of the formula
        self.formula_parts = []

        # Add decay factor, its suffix is always equal to the module it belongs to
        d_param_name = f'delta_{self.module_index}'
        self.params.append(d_param_name)
        self.formula_parts.append(f' - {d_param_name} * y[{self.module_index}]')

        # Add factor which expresses the change in expression based on temperature
        gamma_param_name = f'gamma_{self.module_index}'
        self.params.append(gamma_param_name)
        self.formula_parts.append(f' + {gamma_param_name} * temp * y[{self.module_index}]')
        # self.formula_string += f' + {gamma_param_name} * temp'

        # Register if the module has been compiled (which speeds up its evaluation)
        self.formula_is_compiled = False

    @property
    def formula_string(self) -> str:
        """Get string of full formula"""
        return ''.join(self.formula_parts)

    def compile_formula(self):
        # Compile string to speed up evaluation
        self.compiled_formula_string = compile(self.formula_string.lstrip(),
                                               "<string>", "eval")
        self.formula_is_compiled = True

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

    @staticmethod
    def generate_hill_activation_term(beta_param_name: str, k_param_name: str,
                                      var_name: str, n: float = 1.):
        return (f'+ ({beta_param_name} * {var_name}**{n}) '
                f'/ ({k_param_name}**{n} + {var_name}**{n})')

    @staticmethod
    def generate_hill_inhibition_term(beta_param_name: str,
                                      k_param_name: str,
                                      var_name: str, n: float = 1.):
        return (f'+ ({beta_param_name} * {k_param_name}**{n}) '
                f'/ ({k_param_name}**{n} + {var_name}**{n})')

    def __repr__(self):
        return f'Formula of {self.module_name}={self.formula_string} ' \
               f'\n nr_params = {self.nr_params}'

    def __call__(self, t: float, y: list[float],
                 params: dict[str, float]) -> float:
        """When called, calculate the outcome of the formula

        :param t: Current timepoint
        :param y: List of expressions of all modules
        :param params: Dict that maps parameter names to their value
        :return: Derivative at time point t,
                  given parameters and expressions of modules.
        """
        # Create dict for all these things
        init_val = {"y": y}
        # Merge all dicts
        local_dict = init_val | params
        if not self.formula_is_compiled:
            self.compile_formula()
        result = eval(self.compiled_formula_string, {}, local_dict)
        return result

    @property
    def nr_params(self):
        """Get the number of parameters"""
        return len(self.params)


