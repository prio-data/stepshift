import string
import unittest
from sklearn.dummy import DummyClassifier
import numpy as np
import pandas as pd

from stepshift import views

class TestViews(unittest.TestCase):
    def test_views_api(self):
        dat = pd.DataFrame(
                np.random.rand(16*8,10),
                index = pd.MultiIndex.from_product((range(16),range(8))),
                columns = list(string.ascii_letters[:10])
                )

        mdl = views.StepshiftedModels(DummyClassifier(),[*range(1,13)],"a")

        mdl.fit(dat)
        preds = mdl.predict(dat)

        self.assertEqual(preds.shape[1], 12)

    def test_discontinuous_idx(self):
        i = pd.MultiIndex.from_tuples([
                (1,1),
                (2,1),
                (3,1),
                (2,2),
            ], names=("time","unit"))

        dat = pd.DataFrame(
                np.random.rand(4,2),
                columns = ["a","b"],
                index = i)
        mdl = views.StepshiftedModels(DummyClassifier(),[1,2],"a")

        mdl.fit(dat)

        preds = mdl.predict(dat)
        self.assertEqual(preds.shape, (10,2))
