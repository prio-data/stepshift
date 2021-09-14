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
        dat.to_parquet("/tmp/dat.parquet")
        preds.to_parquet("/tmp/preds.parquet")

