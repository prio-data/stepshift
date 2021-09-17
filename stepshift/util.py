import numpy as np
import xarray

step_pred_column_name = lambda i: f"step_pred_{i}"

def empty_prediction_array(times, units, steps):
    final_t = max(times)
    steps_extent = max(steps)

    prediction_period = np.linspace(
        final_t + 1,
        final_t + steps_extent,
        steps_extent,
        dtype = int)

    return xarray.DataArray(
            np.full((
                len(times) + steps_extent,
                len(units),
                len(steps)),
                np.NaN),
            dims = ("time","unit","feature"),
            coords = {
                "time": np.concatenate([
                        times,
                        prediction_period,
                        ]),
                "unit": units,
                "feature": [step_pred_column_name(i) for i in steps]
                })

