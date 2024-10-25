import numpy as np
import pandas as pd
from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
from itertools import combinations

GODAG = get_godag("data/resources/go_annotations/go-basic.obo", optional_attrs={'relationship'})
class GeneModuleGoTerms:
    def __init__(self, go_term_df: pd.DataFrame):
        self.go_terms = go_term_df['# GO'].to_list()
        relationships = {'part_of'}
        self.wang_object = SsWang(self.go_terms, GODAG, relationships)
    def overall_wang_similarity(self):
        pairwise_sims = []
        if self.get_nr_go_terms() < 2:
            return np.nan
        for go1, go2 in combinations(self.go_terms, 2):
            # Get all combinations here
            val = self.wang_object.get_sim(go1, go2)
            assert val is not None, f'{go1} and {go2} return {val}'
            pairwise_sims.append(val)
        return np.mean(pairwise_sims)

    def get_nr_go_terms(self):
        return len(self.go_terms)
