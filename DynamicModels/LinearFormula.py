from DynamicModels.FormulaSuperClass import FormulaSuperClass


class LinearFormula(FormulaSuperClass):
    """
    LinearFormula for a single module, can be provided with custom parameter values
    when called.
    """

    def __init__(self, module_name: str, regulator_names: list[str]):
        super().__init__(module_name)
        for regulator in regulator_names:
            self.add_regulator(regulator)
        self.compile_formula()

    def add_regulator(self, regulator: str):
        """Add a regulator (e.g. MODULE2) to the equation. And returns name of
        the new parameter

        :param regulator: Name of regulator module, e.g. MODULE3
        :return: Name of the new parameter
        """
        regulator_index = int(regulator[-1])
        b_param_name = f'beta_{regulator_index}_{self.module_index}'
        self.regulator_names.append(regulator)
        self.params.append(b_param_name)
        # Currently assume linear relationship
        self.formula_parts.append(self.generate_linear_term(
            b_param_name, f'y[{regulator_index}]'))
        self.formula_is_compiled = False
        return b_param_name

    def remove_regulator(self, regulator_to_remove: str):
        """Remove the regulator from the equation. Only works if regulator
        is already present in the formula.

        :param regulator_to_remove: Name of regulator module, e.g. MODULE3
        :return: Name of the new parameter"""
        raise NotImplementedError('Since rewriting this class, '
                                  'this method is broken')
        regulator_index = int(regulator_to_remove[-1])
        b_param_name = f'beta_{regulator_index}_{self.module_index}'
        assert regulator_to_remove in self.regulator_names, \
            'Regulator cannot be removed, it is not present in the current formula'
        self.params.remove(b_param_name)
        self.regulator_names.remove(regulator_to_remove)
        string_to_remove_from_formula = self.generate_linear_term(
            b_param_name, f'y[{regulator_index}]')
        self.formula_string = self.formula_string.replace(
            string_to_remove_from_formula, '')
        self.formula_is_compiled = False
        return True