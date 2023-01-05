class MyFormula:
    """Formula for a single module, can be provided with custom parameter values when called."""
    def __init__(self, module_name: str, regulator_names: list[str], init_i: int):
        self.module = module_name
        self.module_index = int(module_name[-1]) - 1
        self.params = []
        formula_segments = []
        self.next_param_suffix = init_i
        for regulator in regulator_names:
            # TODO create some .add_term() method or something?
            regulator_index = int(regulator[-1]) - 1
            b_param_name = f'beta{self.next_param_suffix}'
            # k_param_name = f'k{self.next_param_suffix}'
            # self.params.extend([b_param_name, k_param_name])
            self.params.append(b_param_name)
            # Currently assume linear
            formula_segments.append(f'{b_param_name} * y[{regulator_index}]')
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
