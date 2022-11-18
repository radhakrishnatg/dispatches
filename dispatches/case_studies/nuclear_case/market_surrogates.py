import numpy as np
import json
import os
import pandas as pd
from tensorflow import keras
from pyomo.common.fileutils import this_file_dir
surrogate_dir = os.path.join(this_file_dir(), "nn_market_surrogates")


def get_output(input=[15, 0.05, 10, 500]):
    # load NN model
    model = keras.models.load_model(os.path.join(surrogate_dir, "keras_dispatch_frequency"))

    # load NN scale parameters
    with open(os.path.join(surrogate_dir, "dispatch_input_output_scaling_data.json"), 'rb') as f:
        NN_param = json.load(f) 

    # scale data
    xm = NN_param['xm_inputs']
    xstd = NN_param['xstd_inputs']
    wsm = NN_param['ws_mean']
    wsstd = NN_param['ws_std']

    # must be 2 dimension array otherwise there be errors.
    x_test = np.array([input])

    x_test_scaled = (x_test - xm)/xstd
    pred_ws = model.predict(x_test_scaled)
    pred_ws_unscaled = pred_ws * wsstd + wsm

    # print(pred_ws_unscaled)
    print(min(pred_ws_unscaled))

    return pred_ws_unscaled


if __name__ == "__main__":
    # load NN model
    model = keras.models.load_model(os.path.join(surrogate_dir, "keras_dispatch_frequency"))

    # load NN scale parameters
    with open(os.path.join(surrogate_dir, "dispatch_input_output_scaling_data.json"), 'rb') as f:
        NN_param = json.load(f) 

    # scale data
    xm = NN_param['xm_inputs']
    xstd = NN_param['xstd_inputs']
    wsm = NN_param['ws_mean']
    wsstd = NN_param['ws_std']

    data = {"price": [], "frac": [], "min_freq": [], "sum_freq": []}

    # must be 2 dimension array otherwise there be errors.
    for v1 in [15, 20, 25, 30, 35, 40]:
        for v2 in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:

            x_test = np.array([[v1, v2, 10, 500]])

            x_test_scaled = (x_test - xm)/xstd
            pred_ws = model.predict(x_test_scaled)
            pred_ws_unscaled = pred_ws * wsstd + wsm

            data["price"].append(v1)
            data["frac"].append(v2)
            data["min_freq"].append(min(min(pred_ws_unscaled)))
            data["sum_freq"].append(sum(sum(pred_ws_unscaled)))

    data_df = pd.DataFrame(data)
    print(data_df)
