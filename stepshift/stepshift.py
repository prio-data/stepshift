from typing import Callable, Any, Union, Dict, List, Tuple
from toolz.functoolz import curry
import pandas as pd
import numpy as np

def separate_on_column(
        on: str,
        dataframe: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
    return (dataframe[on], dataframe[[c for c in dataframe.columns if c != on]])

def stepshift(outcomes: pd.DataFrame, inputs: pd.DataFrame, shift: int):
    times = inputs.index.get_level_values(0)
    start, end = times.min(), times.max()
    return outcomes.loc[(start + shift)-1:, :], inputs.loc[:end - shift, :]

def time_unit_feature_cube(dataframe):
    span = lambda a: (a.min(),a.max())

    xmin,xmax = span(dataframe.index.get_level_values(0))
    ymin,ymax = span(dataframe.index.get_level_values(1))

    projected_index = lambda idx, i: int(i - idx)-1
    xindex, yindex = map(curry(projected_index), (xmin, ymin))

    size = (
            xmax - xmin,
            ymax - ymin,
            len(dataframe.columns)
        )

    cube = np.full(size, np.NaN)

    for _,row in dataframe.reset_index().iterrows():
        cube[xindex(row[0]), yindex(row[1]), :] = row[2:]

    return cube
