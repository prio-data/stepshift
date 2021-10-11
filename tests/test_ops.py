
import unittest
import numpy as np
from stepshift import ops

class TestOps(unittest.TestCase):
    def test_rowwise_nonmissing(self):
        X,Y = [np.repeat(np.linspace(1,10,10),10).reshape(10,10) for _ in range(2)]
        x,y = ops.rowwise_nonmissing([X,Y]).value

        np.testing.assert_array_equal(X,x)
        np.testing.assert_array_equal(Y,y)
        np.testing.assert_array_equal(x,y)

        np.testing.assert_array_equal(x[:,0],np.linspace(1,10,10))

        X[1,:] = np.NaN
        Y[8,:] = np.NaN

        x,y = ops.rowwise_nonmissing([X,Y]).value
        self.assertEqual(x.shape[0],y.shape[0], 8)

        np.testing.assert_array_equal(
                x[:,0],
                np.array([1,3,4,5,6,7,8,10])
                )

        self.assertTrue(ops.rowwise_nonmissing([
                np.ones((3,3)),
                np.ones((4,4))
                ]).is_nothing())
