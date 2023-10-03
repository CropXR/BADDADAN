from dataclasses import dataclass
from typing import Callable


@dataclass
class OdeLocalParameters:
    """
    Object that is used to parse local parameters to OdeModels.
    Local parameters are parameters that are not the same between multiple datasets.
    """
    u_t: Callable
