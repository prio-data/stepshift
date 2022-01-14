import string
import unittest
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
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
        preds = mdl.predict(dat, combine = False)

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

        preds = mdl.predict(dat, combine = False)
        self.assertEqual(preds.shape, (10,2))

    def test_predicts_properly(self):
        """
        In this test, I test whether StepshiftedModels manage to capture
        a simple time-trend. The trend is expressed in this data:
           ┌─────────────────────────────┐
          a–0   0   0   1   1   0   0   1│
           │                             │
          b–0   0   0   0   1   1   0   1│
           │                             │
          c–0   0   0   0   0   1   1   1│
           └|───|───|───|───|───|───|───|┘
            1 - 2 - 3 - 4 - 5 - 6 - 7 - 8

        x:time y:unit

        The expected result is that the prediction one-step forward will be
        gt 0 for all units, proving that the model has "learned" that 1s
        follow 1s in 1 time step.
        """

        c = np.array([
                0,0,0,
                0,0,0,
                0,0,0,
                1,0,0,
                1,1,0,
                0,1,1,
                0,0,1,
                1,1,1,
            ]).astype(float)

        index = pd.MultiIndex.from_product((range(1,9),range(1,4)),names=("time","unit"))
        data = pd.DataFrame(np.stack([c,c],axis=1),index=index,columns=["dep","indep"])

        mdl = views.StepshiftedModels(GaussianNB(), [1], "dep")
        mdl.fit(data)
        pred = mdl.predict(data)
        np.testing.assert_array_equal(pred.loc[9,:].values[:,0], np.array([1,1,1]))

    def test_retains_index_names(self):
        data = pd.DataFrame(
                np.random.rand(400,2),
                index = pd.MultiIndex.from_product([range(100), range(4)], names = ["time","unit"]),
                columns = ["dep","indep"])
        model = views.StepshiftedModels(DummyClassifier(),[1,2], "dep")
        model.fit(data)
        preds = model.predict(data)
        print(preds)

        self.assertEqual(preds.index.names, data.index.names)
