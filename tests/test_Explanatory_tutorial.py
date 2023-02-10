from pathlib import Path

from Explanatory_tutorial.retrying_from_scratch import retrying_from_scratch


def test_retrying_from_scratch(my_time_series_expressions):
    retrying_from_scratch(Path('../Explanatory_tutorial').resolve(),
                          my_time_series_expressions)
