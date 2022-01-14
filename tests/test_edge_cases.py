
import logging
from unittest import TestCase
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from stepshift.views import StepshiftedModels

class TestEdgeCases(TestCase):
    """
    TestEdgeCases
    =============

    Tests for various edge cases that might not be very useful for modelling,
    but may be nice to support for testing purposes.
    """

    def test_single_uoa(self):
        data = pd.DataFrame(np.array([
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,1],
                [1,1],
                [1,1],
                [1,1],
                [1,0],
                [0,0],
                [0,0],
                [0,0],
                [0,1],
            ]).astype(float),
            columns = ["dep", "a"],
            index = pd.MultiIndex.from_product([range(14), [1,]]))

        model = StepshiftedModels(LinearRegression(), [1,2,3,4], "dep")
        model.fit(data.loc[0:13])

        # Single prediction point
        preds = model.predict(data.loc[[13]])
        np.testing.assert_almost_equal(
                preds["step_combined"].values,
                np.array([np.NaN, 1.0, .75, .5, .25]).astype(float))

        # Predict over several points in time
        preds = model.predict(data.loc[[10,11,12,13]])

    def test_zero_step(self):
        data = pd.DataFrame(
                np.random.rand(200,2),
                columns = ["dep","indep"],
                index = pd.MultiIndex.from_product((range(100),[1,2])))
        model = StepshiftedModels(LinearRegression(), [0], "dep")
        model.fit(data.loc[:50])

        regular_model = LinearRegression()
        regular_model.fit(data.loc[:50]["indep"].values[:, np.newaxis], data.loc[:50]["dep"].values[:, np.newaxis])

        np.testing.assert_almost_equal(
                model.predict(data.loc[50:])["step_pred_0"],
                regular_model.predict(data.loc[50:]["indep"].values[:, np.newaxis]).squeeze())
