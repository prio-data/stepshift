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

    gen = zip(steps,stepshifted(outcomes,inputs,steps,n_units))
    return {step: fn(*shifted) for step,shifted in gen}

def stepshifted(outcomes,inputs,steps,stepsize):
    for step in steps:
        yield outcomes[((step+1)*stepsize)-stepsize:], inputs[:-(step*stepsize),:]
