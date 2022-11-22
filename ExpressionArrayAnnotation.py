from pathlib import Path

import pandas as pd


class ExpressionArrayAnnotation:
    """Converts names of probe on expression array to gene IDs/names

    Example download: https://www.arabidopsis.org/download_files/Microarrays/Affymetrix/AFFY_ATH1_array_elements.txt
    """
    def __init__(self, some_path: Path):
        # Line 4302 contained an error, so just skip it
        # TODO make more dynamic?
        self.df = pd.read_csv(some_path, sep='\t', skiprows=[4301], header=0)

    def probe_to_agi(self, probe_name: str):
        """Takes affymetrix probe name, and returns name of locus name for TAIR

        :param probe_name:
        :return:
        """
        candidate_agi = self.df.loc[self.df.array_element_name == probe_name, 'locus']
        if len(candidate_agi) == 0:
            print(f'{probe_name} NOT FOUND')
            # raise KeyError(f'{probe_name} not found'
            candidate = probe_name
        else:
            candidate = candidate_agi.item()
            # print(f'{probe_name} -> {candidate}')
        return candidate