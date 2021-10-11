
import unittest
import string
import pandas as pd
import numpy as np
from stepshift.stepshift import stepshifted
from stepshift.cast import time_unit_feature_cube, views_format_to_castable

class TestStepshifting(unittest.TestCase):
    def setUp(self):
        units = 100
        times = 600
        features = 3
        self.columns = list(string.ascii_letters[:features])

        rows = units*times

        self.data = pd.DataFrame(
                np.repeat(np.linspace(1,rows,rows), features).reshape(rows,features),
                index = pd.MultiIndex.from_product((range(times),range(units))),
                columns = self.columns)

    def test_shares_memory(self):
        cube = time_unit_feature_cube(views_format_to_castable(self.data))
        steps = stepshifted(self.columns[0], [1,2,3], cube)
        for _,dep,indep in steps:

            for mat in (dep,indep):
                self.assertTrue(np.may_share_memory(mat.data, cube.data))

    def test_stepshifting(self):
        ss = stepshifted(
                self.columns[2],
                [2,4,6,8],
                time_unit_feature_cube(views_format_to_castable(self.data)))

        steps = [*ss]

        self.assertTrue(all([step[1].shape[0] == step[2].shape[0] for step in steps]))
        self.assertTrue(all([step[1].shape[1] == step[2].shape[1] for step in steps]))
