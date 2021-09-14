"""
APIs used by the ViEWS team.
"""
from typing import List
from toolz.functoolz import compose
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

    def fit(self, data):
        cube = self._cast_views_to_tuf(data)
        for step, dep, indep in stepshift.stepshifted(self._outcome, self._steps, cube):
            dep,indep = [cast.stack_time_unit_feature_cube(xa).data for xa in (dep,indep)]
            dep = dep.reshape(dep.shape[0],1)
            dep_indep = ops.rowwise_nonmissing([dep,indep])

            try:
                assert dep_indep.is_just
            except AssertionError:
                raise ValueError("Dependent and independent arrays had differing number of rows")

            self._models[step] = clone(self._base_clf).fit(*reversed([*dep_indep.value]))

    def predict(self, data):
        cube = self._cast_views_to_tuf(data)
        t_min, t_max = (f(cube.coords["time"].data) for f in (lambda d: d.min(), lambda d: d.max()))

        predictions = util.empty_prediction_array(
                cube.coords["time"].data,
                cube.coords["unit"].data,
                self._steps)

        indep = cube[:,:,1:].stack(row=("time","unit")).transpose()
        i = -1 
        for step,model in self._models.items():
            i += 1

            starts_at,ends_at = (t + step for t in (t_min, t_max))

            model_preds = (model.predict(indep)
                           .reshape((ends_at-starts_at)+1, predictions.shape[1])
                       )

            predictions.loc[starts_at:ends_at,:,:][:,:,i] = model_preds

        return self._cast_tuf_to_views(predictions)

    _cast_views_to_tuf = staticmethod(compose(
        cast.time_unit_feature_cube,
        cast.views_format_to_castable))

    _cast_tuf_to_views = staticmethod(cast.tuf_cube_as_dataframe)
