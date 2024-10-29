import numpy as np
import pandas as pd
from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
from itertools import combinations





class EnrichedGeneModuleGoTerms:
    GODAG = get_godag("data/resources/go_annotations/go-basic.obo",
                      optional_attrs={'relationship'})
    """Class to store the enriched GO terms of a module and calculate
     their semantic similarity
    """
    def __init__(self, go_term_df: pd.DataFrame):
        """
        :param go_term_df: Dataframe that contains enriched GO terms.
        Typically output of the find_enrichment.py function of GOA tools.
        """
        self.go_terms = go_term_df['# GO'].to_list()
        relationships = {'part_of'}
        self.wang_object = SsWang(self.go_terms, self.GODAG, relationships)
    def overall_wang_similarity(self):
        """Calculate the mean pairwise wang similarity between all GO terms.

        Used to assess if they provide a 'coherent' description of
        the biological function.
        """
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
        """Get number of enriched GO terms (i.e. length of DF)"""
        return len(self.go_terms)
