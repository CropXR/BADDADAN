import operator

from lmfit import Parameters
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import seaborn as sns

from DynamicModels.OdeFitter import OdeFitter, OdeFitterMultipleDatasets
from helpers import calculate_parameter_distance


def fit_multiple_fitters(fitters: list[OdeFitter | OdeFitterMultipleDatasets],
                         nr_iters: int = None,
                         extra_analysis: bool = False,
                         gt_params: Parameters = None) -> OdeFitter | OdeFitterMultipleDatasets:
    """Fit a list of fitters simultaneously, and return the best fit

    :param fitters: list of OdeFitter instances. Length of list determines
     how many OdeFitters are run in parallel. E.g. 5 items in list means
     5 parallel processes.
    :return: OdeFitter with best parameters
    """
    caller = operator.methodcaller('fit', nr_iters)
    nr_starts = len(fitters)
    with Pool(nr_starts) as p:
        all_fits = p.map(caller, fitters)

    if extra_analysis:
        chi_squares = [fit.chisqr for fit in all_fits]
        sns.histplot(chi_squares, log_scale=True)
        plt.show()
        if gt_params:
            # Calculate parameter distance
            parameter_distances = [calculate_parameter_distance(fit.params, gt_params)
                                   for fit in all_fits]
            sns.scatterplot(y=chi_squares, x=parameter_distances)
            plt.xlabel('parameter_distances')
        else:
            sns.stripplot(y=chi_squares)
        plt.ylabel('chi_square')
        plt.yscale('log')
        plt.show()

    # Select best fit based on bic
    bic_values = [sol.bic for sol in all_fits]
    lowest_bic = min(bic_values)
    best_index = bic_values.index(lowest_bic)
    # If stop is in error message, the optimisation stopped after
    # a max number of iterations, so we can still just use it.
    assert (all_fits[best_index].success
            or 'STOP' in all_fits[best_index].message),\
        f'Best fit was failed fit :( \n{all_fits[best_index].message}'

    # Set the correct parameters based on the best fit; OdeFitter instances
    # are not changed inplace.
    best_fit = fitters[best_index]
    if hasattr(best_fit, 'params'):
        best_fit.params = all_fits[best_index].params
    elif hasattr(best_fit, 'master_params'):
        best_fit.master_params = all_fits[best_index].params
    best_fit.has_been_fitted = True
    return best_fit
