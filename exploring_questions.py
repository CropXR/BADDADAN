import logging
from copy import copy

import numpy as np
from lmfit import fit_report, Parameters, Parameter

from DynamicModels.OdeFitter import OdeFitter
from DynamicModels.OdeModel import OdeModel
from Expressions.ExpressionMatrix import ExpressionMatrixTimeSeries
from helpers import plot_y_and_y_hat


def compare_annotations(soft_path, csv_path):
    """See if gene annotations differ between doing the SOFT-based annotation
    or annotation_file based annotatoin

    :param soft_path: path to soft geo input
    :param csv_path: path to annotation
    :return:
    """
    ...
