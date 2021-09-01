
import string
from unittest import TestCase
import pandas as pd
import numpy as np
import stepshift

row_prog_array = lambda x: (np.linspace(1,x**2,x**2).reshape(x,x)+(x-1)) // x

class TestStepShifting(TestCase):
    def test_basic_stepshifting(self):
        dataframe = pd.DataFrame(row_prog_array(16))
        dataframe.index = pd.MultiIndex.from_product((range(4),range(4)))
        dataframe.columns = list(string.ascii_letters[:16])

        o,i = stepshift.separate_on_column("a",dataframe)
        o,i = stepshift.stepshift(o, i, 1)
        print(o)
        print(i)

        self.assertFalse(False)
