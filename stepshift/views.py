"""
APIs used by the ViEWS team.
"""
from typing import List
from toolz.functoolz import compose
import xarray as xa
import numpy as np
import pandas as pd
from sklearn.base import clone
from stepshift import stepshift, cast, util, ops

class StepshiftedModels():
    """
    A battery of stepshifted models, trained by  the classifier to
    shifted data for each step. For n number of distinct steps (not necessarily
    contiguous):
    """

    def __init__(self, clf, steps: List[int], outcome: str):
        self._base_clf = clf
        self._steps = steps
        self._steps_extent = max(steps)
        self._outcome = outcome
        self._models = dict()
        self._independent_variables = None

    def fit(self, data):
        """
        Fits one model per step, shifting the data in time appropriately.
        """
        self._independent_variables = [c for c in data.columns if c != self._outcome]

        cube = self._cast_views_to_tuf(data)
        for step, dep, indep in stepshift.stepshifted(self._outcome, self._steps, cube):
            dep,indep = [cast.stack_time_unit_feature_cube(xa).data for xa in (dep,indep)]
            dep = dep.reshape(dep.shape[0],1)
            dep_indep = ops.rowwise_nonmissing([dep,indep])

            try:
                assert dep_indep.is_just
            except AssertionError:
                raise ValueError("Dependent and independent arrays had differing number of rows")

            dep,indep = dep_indep.value
            self._models[step] = clone(self._base_clf).fit(indep,dep.squeeze())

    def predict(self, data):
        """
        Uses the trained models to create a dataset of predictions.
        """
        data = data.sort_index(level = [1,0])

        preds = util.empty_prediction_array(
                np.unique(data.index.get_level_values(0)),
                np.unique(data.index.get_level_values(1)),
                self._steps)

        raw_idx = np.array([np.array(i) for i in data.index.values])

        i = -1
        for step,model in self._models.items():
            i += 1

            raw_predictions = model.predict(data[self._independent_variables].values)

            mat = np.stack([
                        raw_idx[:,0]+step,
                        raw_idx[:,1],
                        raw_predictions,
                    ], axis = 1)

            cube = cast.time_unit_feature_cube(
                    xa.DataArray(mat, dims = ("rows","features"))
                    )

            pred_start,pred_end = [fn(cube.coords["time"]) for fn in (min,max)]
            feature_columns = preds.coords["feature"][i]
            preds.loc[pred_start:pred_end, :, feature_columns] = cube.data.squeeze()

        return self._cast_tuf_to_views(preds)

    _cast_views_to_tuf = staticmethod(compose(
        cast.time_unit_feature_cube,
        cast.views_format_to_castable))

    _cast_tuf_to_views = staticmethod(cast.tuf_cube_as_dataframe)
