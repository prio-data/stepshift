
from operator import and_, eq
from pymonad.maybe import Maybe, Just, Nothing
import numpy as np
from toolz.functoolz import curry, reduce

nonmissing = lambda ax,arr: np.isfinite(arr).all(axis=ax)

def common_nonmissing(ax, arrays)-> Maybe[np.ndarray]:
    if not reduce(eq, map(lambda a: a.shape[0], arrays)):
        return Nothing

    mask = reduce(and_, map(curry(nonmissing, ax), arrays))
    return Just(map(lambda a: a[mask,:], arrays))

rowwise_nonmissing = curry(common_nonmissing, 1)
