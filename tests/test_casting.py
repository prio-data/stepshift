
import string
import unittest
import pandas as pd
import numpy as np
import xarray
from toolz.functoolz import compose
from stepshift import cast

cast_from_views = compose(
        cast.time_unit_feature_cube,
        cast.views_format_to_castable
    )

class TestCasting(unittest.TestCase):
    def test_tuf_cube(self):
        data = pd.DataFrame(
                np.repeat(np.linspace(1,16,16),3).reshape(16,3),
                index = pd.MultiIndex.from_product((range(4),range(4))),
                columns = list(string.ascii_letters[:3]),
                )

        cube:xarray.DataArray = cast_from_views(data)

        self.assertEqual(len(cube.indexes["time"]),4)
        self.assertEqual(len(cube.indexes["unit"]),4)
        self.assertEqual(len(cube.indexes["feature"]),3)

        np.testing.assert_array_equal(
                cube.coords["feature"],
                np.array(["a","b","c"])
                )

        np.testing.assert_array_equal(
                cube.loc[:,0,"a"],
                np.array([1,5,9,13])
                )

        np.testing.assert_array_equal(
                cube.loc[0,:,"a"],
                np.array([1,2,3,4])
                )

    def test_discontinuous(self):
        data = pd.DataFrame(
                np.repeat(np.linspace(1,9,9),3).reshape(9,3),
                index = pd.MultiIndex.from_product([range(3),[1,10,3]]),
                columns = list(string.ascii_letters[:3])
                )

        cube:xarray.DataArray = cast_from_views(data)

        np.testing.assert_array_equal(
                cube.loc[0,[1,10,3],"a"],
                np.array([1,2,3])
                )

        data = pd.DataFrame(
                np.repeat(np.linspace(1,9,9),3).reshape(9,3),
                index = pd.MultiIndex.from_product([range(3),[1,99,150]]),
                columns = list(string.ascii_letters[:3])
                )

        cube:xarray.DataArray = cast_from_views(data)

        np.testing.assert_array_equal(
                cube.loc[0,:,"a"],
                np.array([1,2,3])
                )
