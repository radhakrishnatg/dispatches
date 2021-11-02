import pandas as pd
import numpy as np
from pyomo.environ import Param, RangeSet


def append_lmp_signal(m,
                      signal_source="ARPA_E",
                      signal_name="MiNg_$100_CAISO"):
    if signal_source == "ARPA_E":
        raw_data = pd.read_excel(
            "FLECCS_Price_Series_Data_01_20_2021.xlsx",
            sheet_name="2035 - NREL")

        price_all = raw_data[signal_name].tolist()

    elif signal_source == "RTS_GMLC":
        raw_data = np.load("rts_results_all_prices.npy")
        price_all = raw_data.tolist()

    lmp_data_full_year = np.array([price_all[24 * i: 24 * (i + 1)]
                                   for i in range(364)])

    # m.set_days = RangeSet(lmp_data_full_year.shape[0])
    num_days = 364
    m.set_days = RangeSet(num_days)
    m.set_hours = RangeSet(24)

    m.weights_days = Param(m.set_days, initialize={
        d: (364 / num_days) for d in m.set_days})

    m.LMP = Param(m.set_hours, m.set_days,
                  initialize={
                      (t, d): lmp_data_full_year[d - 1, t - 1]
                      for t in m.set_hours for d in m.set_days})
