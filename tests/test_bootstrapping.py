
from unittest import TestCase
import pandas as pd
import numpy as np
from stepshift import index_draws

class TestBootstrapping(TestCase):
    def test_bootstrapping(self):
        data = pd.DataFrame(np.zeros((9,1)))
        data.index = pd.MultiIndex.from_product((range(3),range(3)))
        draws = index_draws(0,1,data).value

        for d in index_draws(0,1,data).value:
            self.assertEqual(d.sum(),3)

        for d in index_draws(1,1,data).value:
            self.assertEqual(d.sum(),3)
