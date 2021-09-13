"""
APIs used by the ViEWS team.
"""
from typing import List
from toolz.functoolz import compose
import numpy as np
import xarray
from sklearn.base import clone
from stepshift import stepshift, cast, util

class StepshiftedModels():
    """
    A battery of stepshifted models, trained by fitting the classifier to
    shifted data for each step. For n number of distinct steps (not necessarily
    contiguous):
    """

    def __init__(self, clf, steps: List[int], outcome: str):
        self._base_clf = clf
        self._steps = steps
        self._steps_extent = max(steps)
        self._outcome = outcome
        self._models = dict()

    def fit(self, data):
        cube = self._cast_views_to_tuf(data)
        for step, dep, indep in stepshift.stepshifted(self._outcome, self._steps, cube):
            dep, indep = (cast.stack_time_unit_feature_cube(cb) for cb in (dep, indep))
            self._models[step] = clone(self._base_clf).fit(indep, dep)

    def predict(self, data):
        cube = self._cast_views_to_tuf(data)
        t_min, t_max = (f(cube.coords["time"].data) for f in (lambda d: d.min(), lambda d: d.max()))

        predictions = util.empty_prediction_array(
                data.coords["time"].data,
                data.coords["unit"].data,
                self._steps)

        indep = cube.loc[:,:,1:]
        for idx,items in enumerate(self._models.items()):
            step,model = items

            model_preds = (model
                           .predict(indep.stack(row=("time","unit")).transpose())
                           .reshape(indep.shape[0], indep.shape[1])
                       )

            starts_at,ends_at = (t + step for t in (t_min, t_max))
            predictions.loc[starts_at:ends_at,:,:][:,:,idx] = model_preds

        return self._cast_tuf_to_views(predictions)

    _cast_views_to_tuf = staticmethod(compose(
        cast.time_unit_feature_cube,
        cast.views_format_to_castable))

    _cast_tuf_to_views = staticmethod(cast.tuf_cube_as_dataframe)
