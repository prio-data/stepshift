cimport numpy as c_np
import numpy as np

cdef (int, int) minmax(double[:] arr):
    cdef double min = np.inf
    cdef double max = -np.inf
    cdef int i
    for i in range(arr.shape[0]):
        if arr[i] < min:
            min = arr[i]
        if arr[i] > max:
            max = arr[i]
    return <int> min,<int> max

cdef (int) span(int a, int b):
    return b - a

cdef (int) find_index(int idx, c_np.int_t[:] arr):
    cdef int i
    i = 0 
    for i in range(arr.shape[0]):
        if arr[i] == idx:
            break
    return i

def time_unit_feature_cube(c_np.ndarray[double, ndim=2] raw_matrix):
    """
    This function casts a matrix into a time_unit_feature cube,
    which is a useful transformation when indexing in the
    time dimension.

    Expects an matrix, where the first column holds the TIME index,
    and the second column holds the UNIT index. The remaining columns
    are assumed to be FEATURES.
    """

    # First figure out how to compute time-indices (min - i)
    time_min,time_max = minmax(raw_matrix[:,0])
    def time_index(int i):
        return <int> (time_min - i)

    # Then, figure out how to compute unit indices
    # i is equal to the index of the unit in an ordered array of the unique
    # unit identifiers.
    cdef c_np.int_t[:] unique_units
    unique_units = np.unique(raw_matrix[:,1]).flatten().astype(int)
    lookup = dict()
    def unit_index(int i):
        if i in lookup.keys():
            return lookup[i]
        idx = find_index(i, unique_units)
        lookup[i] = idx
        return idx

    # Create an empty cube for holding the result
    cdef c_np.ndarray[double, ndim=3] result_array
    result_array = np.full([
            (time_max - time_min)+1,
            len(unique_units), 
            (raw_matrix.shape[1]-2)
            ], np.inf, dtype = np.float64) 

    # Then iterate over each row in the original dataset, adding the
    # values to the new cube
    cdef int i
    cdef double[:] row
    for i in range(raw_matrix.shape[0]):
        row = raw_matrix[i,:]
        result_array[time_index(<int> row[0]), unit_index(<int> row[1]), :] = row[2:] 

    return result_array 
