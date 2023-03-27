import operator

from pathos.multiprocessing import ProcessingPool as Pool

from DynamicModels.OdeFitter import OdeFitter


def fit_multiple_fitters(fitters: list[OdeFitter]) -> OdeFitter:
    """Fit a list of fitters simultaneously, and return the best fit

    :param fitters: list of OdeFitter instances. Length of list determines
     how many OdeFitters are run in parallel. E.g. 5 items in list means
     5 paralllel processes.
    :return: OdeFitter with best parameters
    """
    caller = operator.methodcaller('fit')
    nr_starts = len(fitters)
    with Pool(nr_starts) as p:
        all_fits = p.map(caller, fitters)
    # Select best fit based on bic
    bic_values = [sol.bic for sol in all_fits]
    lowest_bic = min(bic_values)
    best_index = bic_values.index(lowest_bic)
    assert all_fits[best_index].success, f'Best fit was failed fit :(' \
                                         f'\n{all_fits[best_index].message}'
    # Set the correct parameters based on the best fit; OdeFitter instances
    # are not changed inplace.
    best_fit = fitters[best_index]
    best_fit.params = all_fits[best_index].params
    best_fit.has_been_fitted = True
    return fitters[best_index]
