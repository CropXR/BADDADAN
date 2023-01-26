from pathlib import Path

import numpy as np
import pandas as pd
import logging


class ExpressionArrayAnnotation:
    """Converts names of probe on expression array to gene IDs/names

    Example download: https://www.arabidopsis.org/download_files/Microarrays/Affymetrix/affy_ATH1_array_elements-2010-12-20.txt
    """
    def __init__(self, some_path: Path):
        self.df = pd.read_csv(some_path, sep='\t', header=0)

    def probe_to_agi(self, probe_name: str, verbose: bool = False) -> str:
        """Takes affymetrix probe name, and returns name of locus name for TAIR

        :param probe_name: Name of probe name in microarray, e.g. 263102_at
        :param verbose: If true, print all probes that can not be assigned an
                        agi identifier.
        :return: AGI identifier if found, else original probe name
        """
        # For now just keep multiple annotations per probe?
        candidate_agi = self.df.loc[self.df.array_element_name == probe_name,
                                    'locus']
        if len(candidate_agi) == 0 or np.any(candidate_agi == 'no_match'):
            # Could not find annotation in database
            logging.debug(f'Could not find annotation of {probe_name} in database, proceeding with original name.')
            candidate = probe_name
        else:
            candidate = candidate_agi.item()
        logging.debug(f'{probe_name} -> {candidate}')
        return candidate
