import numpy as np
from numpy.linalg import norm
from numba import njit
import joblib

import matplotlib.pyplot as plt


if __name__ == "__main__":
    evoked = joblib.load("evokeds/sub-CC720670/evoked_fixed_full.pkl")
    signal = evoked.data

    global_field_power = norm(signal, axis=-1)
    idx_max_intensity = np.argmax(global_field_power)
    max_intensity = np.max(global_field_power)

    fig = plt.figure()
    plt.plot(global_field_power)

    left_bound = np.where(
        global_field_power[:idx_max_intensity][::-1] < max_intensity / 2
    )[0][0]

    right_bound = np.where(
        global_field_power[idx_max_intensity:] < max_intensity / 2
    )[0][0]

    plt.axvline(idx_max_intensity, linestyle="--", c="r", linewidth=1)

    plt.axvline(
        idx_max_intensity - left_bound,
        linestyle="-",
        c="r",
        linewidth=1,
    )
    plt.axvline(
        idx_max_intensity + right_bound,
        linestyle="-",
        c="r",
        linewidth=1,
    )

    plt.show()
