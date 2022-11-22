from pathlib import Path

import pytest

from ExpressionArrayAnnotation import ExpressionArrayAnnotation
from ExpressionMatrix import ExpressionMatrix

@pytest.fixture
def my_expression_annotation():
    my_path = Path('/home/bnoordijk/phd/sandbox_gene_expression/AFFY_ATH1_array_elements.txt')
    return ExpressionArrayAnnotation(my_path)

@pytest.fixture
def my_expression_matrix(my_expression_annotation):
    my_expression = Path('/home/bnoordijk/phd/sandbox_gene_expression/GSE15689_family.soft')
    expr_mat = ExpressionMatrix.from_geo_file(my_expression, my_expression_annotation)
    return expr_mat