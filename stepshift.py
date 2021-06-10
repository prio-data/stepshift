from typing import Callable, Any, Union, Dict, List
import pandas as pd
import numpy as np

def stepshift(
        dataframe: pd.DataFrame,
        fn: Callable[[np.ndarray, np.ndarray], Any],
        steps: List[int],
        outcome: Union[str,int] = 0,
        )-> Dict[int,Any]:
    if isinstance(outcome, str):
        outcome = list(dataframe.columns).index(outcome)

    outcomes = dataframe.values[:, outcome]
    inputs = np.delete(dataframe.values, outcome, 1)

    try:
        n_units = len({u for t,u in dataframe.index.values})
    except ValueError:
        raise TypeError("Could not unpack index. Expected a multiindex of length 2")

    return {s: fn(outcomes, inputs[(s* n_units) - n_units:, :]) for s in steps}
