"""Stepshift
from typing import List

Stepshifting is a procedure that is used when training models to predict future
values of a dependent variable.

In practice, this involves training models on data that is shifted in time,
relative to the dependent variable. Here is an example with step size 1:

 ──────────────
 Dependent
 ┌─┐┌─┐┌─┐
 │0││1││2│
 └─┘└─┘└─┘
 ---
 Independent(s)
    ^^^^^^---
    ┌─┐┌─┐┌─┐
  ─►│0││1││2│
    └─┘└─┘└─┘
    ┌─┐┌─┐┌─┐
  ─►│0││1││2│
    └─┘└─┘└─┘
    ┌─┐┌─┐┌─┐
  ─►│0││1││2│
    └─┘└─┘└─┘
 ──────────────
 --- = masked (subset)
 ^^^ = predicts

The step size determines how far into the future the resulting model will be
able to predict.
"""

from typing import List, Generator, Tuple
import logging
import xarray

logger = logging.getLogger(__name__)

def set_feature_as_first(x:str, array: xarray.DataArray)-> xarray.DataArray:
    try:
        assert x in array.coords["feature"]
    except AssertionError:
        raise ValueError(f"Feature {x} is not a feature in array")

    wanted_order = [x] + [i for i in array.coords["feature"].data if i != x]

    if wanted_order == list(array.coords["feature"].data):
        return array
    else:
        logger.warning("Reordering feature dimension. "
                "Save memory by setting the outcome feature as the first column "
                "in your dataframe."
                )
        return array.loc[:,:, wanted_order]

def stepshifted(
        outcome: str,
        steps: List[int],
        array: xarray.DataArray
        )-> Generator[Tuple[int, xarray.DataArray, xarray.DataArray], None, None]:
    """
    Generator which yields a tuple of outcomes and inputs for each time-shift step.
    """
    array = set_feature_as_first(outcome, array)
    outcomes = array[:,:,0]
    inputs = array[:,:,1:]
    for step in steps:
        yield step, outcomes[step:,:], inputs[:-step,:,:]
