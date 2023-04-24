from pathlib import Path

from helpers import extract_flor_id_genes


def test_extract_flor_id_genes():
    extract_flor_id_genes(Path('../data/resources/flor_id_flowering_genes_2.html'),
                          Path('../data/resources/flor_id_flowering_genes.pkl'),)
    return True
