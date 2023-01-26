import logging
from pathlib import Path

import GEOparse
import pandas as pd
import typer

from ExpressionArrayAnnotation import ExpressionArrayAnnotation
from ExpressionMatrix import ExpressionMatrix
from predict_from_static_expressions import plot_pred_vs_real

pd.options.display.width = 0
GEOparse.logger.set_verbosity('INFO')
logging.basicConfig(level=logging.INFO)


def main(
        expression_path: Path = typer.Option(...,
                                             help='Path to geo expression file'),
        annotation_path: Path = typer.Option(...,
                                             help='Path to annotation of micro array'),
        n_cluster: int = typer.Option(5,
                                      help='Number of gene clusters to '
                                           'extract'),
):
    expression_annotation = ExpressionArrayAnnotation(annotation_path)
    expression_matrix = ExpressionMatrix.from_geo_file(expression_path,
                                                       expression_annotation)
    expression_matrix = expression_matrix.get_only_wt_samples()
    plot_pred_vs_real(expression_matrix, n_cluster)
    # do_cv_for_nclust(expression_matrix)


if __name__ == "__main__":
    typer.run(main)

