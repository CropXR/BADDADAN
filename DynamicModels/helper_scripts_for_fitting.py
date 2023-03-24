import operator

from pathos.multiprocessing import ProcessingPool as Pool

from DynamicModels.OdeFitter import OdeFitter


def fit_multiple_fitters(fitters: list[OdeFitter]) -> OdeFitter:
    caller = operator.methodcaller('fit')
    nr_starts = len(fitters)
    with Pool(nr_starts) as p:
        all_fits = p.map(caller, fitters)
    # Select best fit based on bic
    bic_values = [sol.bic for sol in all_fits]
    lowest_bic = min(bic_values)
    best_index = bic_values.index(lowest_bic)
    return fitters[best_index]
