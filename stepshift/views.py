"""
APIs used by the ViEWS team.
"""
import re
from typing import List, Optional, Dict, Literal
from toolz.functoolz import compose, curry
from pymonad.maybe import Just, Nothing
import xarray as xa
import numpy as np
import pandas as pd
from sklearn.base import clone
from stepshift import stepshift, cast, util, ops

def step_combine(
        predictions: pd.DataFrame,
        column_step_mapping: Optional[Dict[str,int]] = None) -> pd.DataFrame:

    units = np.unique(predictions.index.get_level_values(1).values)

    if column_step_mapping is None:
        column_step_mapping = infer_column_step_mapping(predictions.columns)

    step_size = max(column_step_mapping.values())
    times = predictions.index.get_level_values(0)
    pred_start,pred_end = (fn(times) for fn in (min,max))
    pred_period_size = (pred_end-pred_start)+1

    data = np.stack([np.full(pred_period_size, np.NaN)]*len(units),axis=1)
    step_combined = xa.DataArray(
            data,
            dims = ("time","unit"),
            coords = {
                "time": np.unique(times),
                "unit":units})

    sc_period_start = pred_end - (step_size)
    for step_name, step_value in column_step_mapping.items():
        sc_time = sc_period_start+step_value
        step_combined.loc[sc_time,:] = predictions[[step_name]].loc[sc_time,:].squeeze()

    return step_combined.stack(step_combined=("time","unit")).data

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
        self._models = {}
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

    def _predict(self, data,
            combine: bool = True,
            kind: Literal["predict", "predict_proba"] = "predict"):
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

            raw_predictions = getattr(model, kind)(data[self._independent_variables].values)

            if len(raw_predictions.shape) == 1:
                pass
            elif len(raw_predictions.shape) == 2:
                raw_predictions = raw_predictions[:,raw_predictions.shape[1]-1]
            else:
                raise ValueError(
                        f"Model produced predictions with {len(raw_predictions.shape)} dims. "
                        "Expected 1 or 2. (Note, multivariate classification is not supported)."
                        )

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

        df = self._cast_tuf_to_views(preds)

        if combine:
            df["step_combined"] = step_combine(df)

        return df

    def predict(self, data, combine: bool = True):
        return self._predict(data, combine, kind = "predict")
    def predict_proba(self, data, combine: bool = True):
        return self._predict(data, combine, kind = "predict_proba")

    _cast_views_to_tuf = staticmethod(compose(
        cast.time_unit_feature_cube,
        cast.views_format_to_castable))

    _cast_tuf_to_views = staticmethod(cast.tuf_cube_as_dataframe)

def infer_column_step_mapping(names: str):
    step_pred_column_name_regex = f"(?<={util.step_pred_column_name('')})[0-9]+"
    step_name_from_column_name = compose(
            lambda m: m.maybe(None,int),
            lambda m: Just(m.group()) if m else Nothing,
            curry(re.search, step_pred_column_name_regex))

    matches = {n: step_name_from_column_name(n) for n in names}
    return {k:v for k,v in matches.items() if v is not None}
