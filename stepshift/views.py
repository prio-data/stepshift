"""
APIs used by the ViEWS team.
"""
import logging
import re
from typing import List, Optional, Dict, Literal
from toolz.functoolz import compose, curry
from pymonad.maybe import Just, Nothing
import xarray as xa
import numpy as np
import pandas as pd
from sklearn.base import clone
from stepshift import stepshift, cast, ops

logger = logging.getLogger(__name__)

class StepshiftedModels():
    """
    StepshiftedModels
    =================

    parameters:
        clf (Classifier):  A scikit-learn classifier
        steps (List[int]): A list of steps for which to train models
        outcome (str):     The name of the column to use as the outcome.

    example:
        models = StepshiftedModels(
                sklearn.linear_model.LogisticRegression(),
                [1,2,3],
                "outcome_column")

    A battery of stepshifted models, trained by  the classifier to shifted data
    for each step. Stepshifting means shifting the independent variables in
    time n steps for each step n, and then training a model predicting the
    dependent variable Y. This produces a set of models trained to predict
    future outcomes, with varying time-scope. Such a set can be used to produce
    a series of future predictions Ŷ where each prediction is the result of a
    model trained specifically to predict that number of steps into the future.

    For further elaboration of stepshifting, see appendix A of Hegre et al. 2020
    (10.1177/0022343320962157)
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
        fit
        ===

        parameters:
            data (pandas.DataFrame): A time-unit indexed dataframe

        example:
            models = StepshiftedModels(
                    sklearn.linear_model.LogisticRegression(),
                    [1,2,3],
                    "y")

            df = pd.DataFrame(
                np.zeros((32 * 32, 2)),
                index = pd.MultiIndex.from_product((range(32), range(32)), names = ("time", "unit")),
                columns = ["y", "x"])

            models.fit(df)

        Fits one model per time-step, to create a battery of models capable of
        predicting future time outcomes.
        """
        self._independent_variables = [c for c in data.columns if c != self._outcome]

        cube = self._cast_views_to_tuf(data)
        for step, dep, indep in stepshift.stepshifted(self._outcome, self._steps, cube):
            dep,indep = [cast.stack_time_unit_feature_cube(xa).data for xa in (dep,indep)]
            dep = dep.reshape(dep.shape[0],1)

            logger.debug(f"Dep. values of shape {dep.shape}")
            logger.debug(f"Indep. values of shape {indep.shape}")
            dep_indep = ops.rowwise_nonmissing([dep,indep])

            try:
                assert dep_indep.is_just()
            except AssertionError:
                raise ValueError("Dependent and independent arrays had differing number of rows")

            dep,indep = dep_indep.value
            logger.debug(f"Saving model {step}")
            self._models[step] = clone(self._base_clf).fit(indep,dep.squeeze())

    def predict(self, data, combine: bool = True):
        """
        predict
        =======

        parameters:
            data (pandas.DataFrame): A time-unit indexed dataframe.
            combine (bool): Create a step-combined column in the output?

        returns:
            pandas.DataFrame (time-unit indexed)

        example:
            models = StepshiftedModels(
                    sklearn.linear_model.LogisticRegression(),
                    [1,2,3],
                    "y")

            df = pd.DataFrame(
                np.zeros((32 * 32, 2)),
                index = pd.MultiIndex.from_product((range(32), range(32)), names = ("time", "unit")),
                columns = ["y", "x"])

            train,test = df.loc[0:12, :], df.loc[13:, :]

            models.fit(train)
            predictions = models.predict(test)

        Creates a dataset of outcome predictions using the predict method of
        the model(s). The combine argument decides whether to include a
        step-combined column in the output. The output has columns named
        "step_pred_{i}" with predictions for each step.
        """
        return self._predict(data, combine, kind = "predict")

    def predict_proba(self, data, combine: bool = True):
        """
        predict_proba
        =======

        parameters:
            data (pandas.DataFrame): A time-unit indexed dataframe.
            combine (bool): Create a step-combined column in the output?

        returns:
            pandas.DataFrame (time-unit indexed)

        example:
            models = StepshiftedModels(
                    sklearn.linear_model.LogisticRegression(),
                    [1,2,3],
                    "y")

            df = pd.DataFrame(
                np.zeros((32 * 32, 2)),
                index = pd.MultiIndex.from_product((range(32), range(32)), names = ("time", "unit")),
                columns = ["y", "x"])

            train,test = df.loc[0:12, :], df.loc[13:, :]

            models.fit(train)
            predictions = models.predict_proba(test)

        Creates a dataset of probability predictions using the predict method
        of the model(s). The combine argument decides whether to include a
        step-combined column in the output.
        """
        return self._predict(data, combine, kind = "predict_proba")

    @staticmethod
    def step_pred_column_name(i: int)-> str:
        return f"step_pred_{i}"

    @property
    def models(self):
        return self._models

    @staticmethod
    def _pre_sort(data: pd.DataFrame)-> pd.DataFrame:
        """
        Ensures data is sorted by UNIT-TIME. Necessary to avoid issues with the stepshifting algorithm.
        """
        return data.sort_index(level = [1,0])

    @staticmethod
    def _ensure_predictions_shape(predictions: np.ndarray)-> np.ndarray:
        """
        Handles output from models, ensuring it is one-dimensional.
        """

        if len(predictions.shape) == 1:
            return predictions
        elif len(predictions.shape) == 2:
            return predictions[:,predictions.shape[1]-1]
        else:
            raise ValueError(
                    f"Model produced predictions with {len(predictions.shape)} dims. "
                    "Expected 1 or 2. (Note, multivariate classification is not supported)."
                    )

    def _empty_prediction_array(self, times: int, units: int, steps: int):
        """
        Create an empty array for storing predictions; a 3rd order tensor
        (cube) with TIME-UNIT-FEATURE dimensions.
        """
        final_t = max(times)
        steps_extent = max(steps)

        prediction_period = np.linspace(
            final_t + 1,
            final_t + steps_extent,
            steps_extent,
            dtype = int)

        return xa.DataArray(
                np.full((
                    len(times) + steps_extent,
                    len(units),
                    len(steps)),
                    np.NaN),
                dims = ("time","unit","feature"),
                coords = {
                    "time": np.concatenate([
                            times,
                            prediction_period,
                            ]),
                    "unit": units,
                    "feature": [self.step_pred_column_name(i) for i in steps]
                    })

    def _predict(self, data,
            combine: bool = True,
            kind: Literal["predict", "predict_proba"] = "predict") -> pd.DataFrame:
        """
        Uses the trained models to create a dataset of predictions.
        """

        data = self._pre_sort(data)
        preexisting_index_names = data.index.names

        preds = self._empty_prediction_array(
                np.unique(data.index.get_level_values(0)),
                np.unique(data.index.get_level_values(1)),
                self._steps)

        raw_idx = np.array([np.array(i) for i in data.index.values])
        for step_column_name, (step, model) in zip(preds.coords["feature"],self._models.items()):
            logger.debug(f"Making predictions for model {step} ({step_column_name.data})")

            raw_predictions = self._ensure_predictions_shape(
                    getattr(model, kind)(data[self._independent_variables].values))

            mat = np.stack([
                        raw_idx[:,0]+step,
                        raw_idx[:,1],
                        raw_predictions,
                    ], axis = 1)

            cube = cast.time_unit_feature_cube(
                    xa.DataArray(mat, dims = ("rows","features"))
                    )

            pred_start, pred_end = [fn(cube.coords["time"]) for fn in (min,max)]
            dat = cube.data.squeeze()


            # Ensure that data is two-dimensional (force 1-sized dimension in second dim if necessary)
            if len(dat.shape) == 0:
                dat = dat[np.newaxis, np.newaxis]
            elif len(dat.shape) == 1:
                dat = dat[: , np.newaxis]
            else:
                pass

            logger.debug(f"Preds of shape {dat.shape}")

            preds.loc[pred_start:pred_end, :, step_column_name] = dat


        df = self._cast_tuf_to_views(preds)

        if combine:
            df["step_combined"] = step_combine(df)

        df.index.names = preexisting_index_names

        return df

    _cast_views_to_tuf = staticmethod(compose(
        cast.time_unit_feature_cube,
        cast.views_format_to_castable))

    _cast_tuf_to_views = staticmethod(cast.tuf_cube_as_dataframe)

def step_combine(
        predictions: pd.DataFrame,
        column_step_mapping: Optional[Dict[str,int]] = None) -> np.ndarray:
    """
    step_combine
    ============

    parameters:
        predictions (pandas.DataFrame):                 A dataframe containing predictions
        column_step_mapping (Optional[Dict[str, int]]): Inferred if not provided, see infer_column_step_mapping

    returns:
        numpy.ndarray:                                  A 2nd order tensor containing the combined predictions

    Combines predictions from the provided dataframe into a 2nd order tensor,
    which can be assigned back into the dataframe to create a step_combined
    column. Used internally by the StepshiftedModels class.
    """

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

def infer_column_step_mapping(names: List[str]):
    """
    infer_column_step_mapping
    =========================

    parameters:
        names (List[str]): A list of column names

    returns:
        Dict[str, int]: Names mapped to step numbers (if applicable).

    """
    step_pred_column_name_regex = f"(?<={StepshiftedModels.step_pred_column_name('')})[0-9]+"
    step_name_from_column_name = compose(
            lambda m: m.maybe(None,int),
            lambda m: Just(m.group()) if m else Nothing,
            curry(re.search, step_pred_column_name_regex))

    matches = {n: step_name_from_column_name(n) for n in names}
    return {k:v for k,v in matches.items() if v is not None}
