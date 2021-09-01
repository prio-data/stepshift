
from typing import Generator, List, Callable, TypeVar
from toolz.functoolz import compose, curry
from pymonad.either import Left, Right, Either
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

T = TypeVar("T")

def index_draws(
        level: int,
        draw_size: int,
        dataframe: pd.DataFrame)-> Either[str, Generator[ArrayLike, None, None]]:
    """index_draws

    Function that returns an Either containing a generator of random draws from
    a dataset. Draws are made at an index level, meaning that a subset of index-
    units is returned.

    Example:

        To demonstrate, let data be a dataframe containing the following rows
        and columns:

        unit*| time*| value
        -------------------
           0 |    0 |     1
           0 |    1 |     2
           0 |    2 |     3
           1 |    0 |     4
           1 |    1 |     5
           1 |    2 |     6
           2 |    0 |     7
           2 |    1 |     8
           2 |    2 |     9

        This dataset may be subset on either index columns (marked by *). If
        level=0 (in this case, unit*), then the generator will yield rows with
        N=draw_size unique values in the unit* column:

        index_draws(level=0, draw_size=2, data)

        unit*| time*| value
        -------------------
           0 |    0 |     1
           0 |    1 |     2
           0 |    2 |     3
           2 |    0 |     7
           2 |    1 |     8
           2 |    2 |     9

        If level=1 (time*), the same operation could yield the following data:

        index_draws(level=1, draw_size=2, data)

        unit*| time*| value
        -------------------
           0 |    0 |     1
           0 |    2 |     3
           1 |    0 |     4
           1 |    2 |     6
           2 |    0 |     7
           2 |    2 |     9
    """

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


def n_times(function: Callable[[pd.DataFrame],T],times: int,data: pd.DataFrame)-> List[T]:
    return map(function(data), range(times))

def bootstrapped(
        subset_function: Callable[[pd.DataFrame], pd.DataFrame],
        draws: int,
        function: Callable[[pd.DataFrame], T],
        data: pd.DataFrame,
        ) -> List[T]:
    """bootstrapped

    Bootstrapped function application. Applies a function to data n_draws times
    """
    return n_times(compose(function,subset_function), draws, data)

def level_bootstrapped(
        level: int,
        draw_size: int,
        draws: int,
        function: Callable[[pd.DataFrame], T],
        data: pd.DataFrame,
        ) -> List[T]:
    return bootstrapped(
            curry(index_draws, level, draw_size),
            draws, function, data
            )
