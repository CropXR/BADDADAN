from pathlib import Path

import numpy as np
import pandas as pd
import logging


class ExpressionArrayAnnotation:
    """Converts names of probe on expression array to gene IDs/names

    Example download: https://www.arabidopsis.org/download_files/Microarrays/Affymetrix/affy_ATH1_array_elements-2010-12-20.txt
    """
    def __init__(self, some_path: Path, sep='\t', array_type: str = 'affy'):
        self.df = pd.read_csv(some_path, sep=sep, header=0)
        if array_type == 'affy':
            self.probe_to_agi = self.affymetrix_conversion
        elif array_type == 'catma':
            self.probe_to_agi = self.catma_conversion
            self.conversion_dict = self.df.set_index('CATMA_ID')['AGI_code_Spring_2004'].to_dict()
        else:
            raise NotImplementedError(f'Annotation from array type {array_type}'
                                      f' cannot be used at the moment.')

        # TODO make this just a dict for speedups

    def affymetrix_conversion(self, probe_name: str, verbose: bool = False) -> str:
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

    def catma_conversion(self, probe_name: str) -> str:
        """For a catma array, use this to convert from probe name to gene ID"""
        if probe_name not in self.conversion_dict:
            logging.warning(f'Did not find {probe_name}. Returning original probe name')
            return probe_name
        return self.conversion_dict[probe_name]
