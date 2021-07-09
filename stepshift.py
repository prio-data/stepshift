from typing import Callable, Any, Union, Dict, List, Generator
from pymonad.either import Left, Right, Either
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

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

def index_draws(
        level: int,
        draw_size: int,
        dataframe: pd.DataFrame)-> Either[str, Generator[ArrayLike, None, None]]:
    def draws(which, from_what)-> Generator[ArrayLike, None, None]:
        for pick in which:
            mask = np.full(len(from_what),False)
            for value in pick:
                mask |= from_what == value
            yield mask

    try:
        idx = np.array([i[level] for i in dataframe.index])
    except TypeError:
        return Left("Dataframe must have a multiindex")
    except KeyError:
        return Left(
                "Dataframe had incorrect number of levels. "
                "Needed level {level}, had "
                "{len(dataframe.index[0])} levels."
            )

    unique_values = list(set(idx))
    picks = np.linspace(0,len(unique_values)-1,len(unique_values))
    np.random.shuffle(picks)

    n_picks = len(picks)
    picks = picks[:n_picks - (n_picks % draw_size)]
    picks = picks.reshape(len(picks) // draw_size, draw_size)

    return Right(draws(picks,idx))

