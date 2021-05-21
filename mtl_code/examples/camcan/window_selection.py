import numpy as np
from numba import njit
import joblib

import matplotlib.pyplot as plt


@njit
def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filtered_y = y  # np.array(y)
    avg_filter = [0] * len(y)
    std_filter = [0] * len(y)
    avg_filter[lag - 1] = np.mean(y[0:lag])
    std_filter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avg_filter[i - 1]) > threshold * std_filter[i - 1]:
            if y[i] > avg_filter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filtered_y[i] = (
                influence * y[i] + (1 - influence) * filtered_y[i - 1]
            )
            avg_filter[i] = np.mean(filtered_y[(i - lag + 1) : i + 1])
            std_filter[i] = np.std(filtered_y[(i - lag + 1) : i + 1])
        else:
            signals[i] = 0
            filtered_y[i] = y[i]
            avg_filter[i] = np.mean(filtered_y[(i - lag + 1) : i + 1])
            std_filter[i] = np.std(filtered_y[(i - lag + 1) : i + 1])

    return signals


if __name__ == "__main__":
    lag = 50
    threshold = 1
    influence = 5

    evoked = joblib.load("evokeds/sub-CC720188/evoked_fixed_full.pkl")

    signal = evoked.data
    avg_over_time = np.mean(signal, axis=-1) * 1e13
    idx_max_signal = np.argsort(avg_over_time)[:5]
    max_signal = signal[idx_max_signal, :].mean(axis=0)
    max_signal = np.diff(max_signal, n=1)

    idx_max_intensity = np.argmax(np.abs(max_signal))

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(max_signal)

    thresholded_signal = thresholding_algo(
        max_signal, lag, threshold, influence
    )
    axes[1].plot(thresholded_signal)

    left_bound = np.where(thresholded_signal[:idx_max_intensity][::-1] == 0)[
        0
    ][0]

    right_bound = np.where(thresholded_signal[idx_max_intensity:] == 0)[0][0]

    axes[0].axvline(idx_max_intensity, linestyle="--", c="r", linewidth=1)
    axes[1].axvline(idx_max_intensity, linestyle="--", c="r", linewidth=1)

    axes[0].axvline(
        idx_max_intensity - left_bound,
        linestyle="-",
        c="r",
        linewidth=1,
    )
    axes[0].axvline(
        idx_max_intensity + right_bound,
        linestyle="-",
        c="r",
        linewidth=1,
    )

    axes[1].axvline(
        idx_max_intensity - left_bound,
        linestyle="-",
        c="r",
        linewidth=1,
    )

    axes[1].axvline(
        idx_max_intensity + right_bound,
        linestyle="-",
        c="r",
        linewidth=1,
    )

    fig.suptitle(f"Lag: {lag}, Thresh: {threshold}, Influence: {influence}")

    plt.show()
