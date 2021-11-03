
import unittest
import hashlib
import random
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from stepshift import views

class TestViewsUtilities(unittest.TestCase):
    def test_col_name_inference(self):
        """
        Checks that step numbers can be casted back and forth into step names,
        resulting in a dictionary which can be used to map names to step numbers.
        """
        step_numbers = [*range(1,11)]
        step_names = [*map(views.StepshiftedModels.step_pred_column_name, step_numbers)]
        inferred_step_numbers = views.infer_column_step_mapping(step_names)

        self.assertEqual([*inferred_step_numbers.values()], step_numbers)
        self.assertEqual([*inferred_step_numbers.keys()], step_names)

    def test_no_bad_names(self):
        """
        Checks that no garbage strings are parsed
        """
        r = random.Random(123)
        garbage = [hashlib.md5(str(r.random()).encode()).hexdigest() for i in range(10)]
        inferred = views.infer_column_step_mapping(garbage)
        self.assertEqual(len(inferred),0)

    def test_step_combined(self):
        """
        Combined steps into a single "step_combined" column.
        """

        n = np.NaN
        mat = np.array([
                [n,n,n,n],
                [n,n,n,n],
                [n,n,n,n],
                [1,n,n,n],
                [1,2,n,n],
                [1,2,3,n],
                [1,2,3,4],
                [n,2,3,4],
                [n,n,3,4],
                [n,n,n,4],
            ])
        data = np.concatenate([mat,mat,mat], axis=0)

        preds = pd.DataFrame(
                data,
                index = pd.MultiIndex.from_product((range(3),range(10)), names=("unit","time")),
                columns = [views.StepshiftedModels.step_pred_column_name(i) for i in range(1,5)]
                ).swaplevel(0,1).sort_index(axis=0)
        preds["step_combined"] = views.step_combine(preds)
        sc_series = preds.query("unit==0")["step_combined"]
        np.testing.assert_array_equal(
                np.array([n,n,n,n,n,n,1,2,3,4],dtype=float),
                sc_series
                )

    def test_auto_combine(self):
        d = pd.DataFrame(
                    np.zeros((16,2)),
                    index = pd.MultiIndex.from_product((range(8),range(2))),
                    columns = list("ab")
                )
        mdl = views.StepshiftedModels(DummyClassifier(),[1,2,3],"a")
        mdl.fit(d)
        preds = mdl.predict(d)
        np.testing.assert_array_equal(
                preds[views.StepshiftedModels.step_pred_column_name(1)].loc[10-2,].values,
                preds["step_combined"].loc[10-2,].values)
