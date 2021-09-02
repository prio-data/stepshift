cimport numpy as c_np
import numpy as np
import pandas as pd

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

def time_unit_feature_cube(dataframe):
    """
    This function casts a matrix into a time_unit_feature cube,
    which is a useful transformation when indexing in the
    time dimension.

    Expects an matrix, where the first column holds the TIME index,
    and the second column holds the UNIT index. The remaining columns
    are assumed to be FEATURES.
    """

    # First some setup
    cdef double[:,:] raw_matrix_view = (dataframe
            .sort_index(level=[1,0]) # The DF is sorted to enable a much 
            .reset_index()           # faster unit-indexing scheme.
            .values
            )

    unique_units = np.unique(raw_matrix_view[:,1])

    # First figure out how to compute time-indices (min - i)
    cdef int time_min
    cdef int time_max 

    time_min,time_max = minmax(raw_matrix_view[:,0])
    def time_index(int i):
        return <int> (time_min - i)

    # Create an empty cube for holding the result
    cdef c_np.ndarray[double, ndim=3] result_array
    result_array = np.full([
            (time_max - time_min)+1,
            len(unique_units), 
            (raw_matrix_view.shape[1]-2)
            ], np.inf, dtype = np.float64) 
    cdef double[:,:,:] result_view = result_array

    # Then iterate over each row in the original dataset, adding the
    # values to the new cube
    cdef int i
    cdef double[:] row
    cdef int t_index
    cdef int u_index = -1 
    cdef int current_u_index = -1 

    for i in range(raw_matrix_view.shape[0]):
        row = raw_matrix_view[i,:]

        if row[1] != current_u_index:
            current_u_index = <int> row[1]
            u_index = u_index + 1

        t_index = time_min - <int> row[0] 
        result_view[t_index, u_index, :] = row[2:] 

    return result_array 
