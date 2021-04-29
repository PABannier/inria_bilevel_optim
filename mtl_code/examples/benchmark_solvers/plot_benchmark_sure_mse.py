import joblib
import numpy as np
import matplotlib.pyplot as plt


def plot_benchmark_synthetic_data(benchmarks, name, annotation):
    x = benchmarks.keys()
    duration_cv = [x[0] for x in benchmarks.values()]
    duration_sure = [x[1] for x in benchmarks.values()]

    fig = plt.figure()

    plt.plot(x, duration_cv, label="5-fold CV")
    plt.plot(x, duration_sure, label="SURE")

    plt.xlabel(f"Number of {name}")
    plt.ylabel("Duration (s)")
    plt.title(annotation)

    plt.legend()

    fig.suptitle(
        f"Hyperopt selection speed vs n. of {name}", fontweight="bold"
    )

    plt.show()


def plot_benchmark_meg_data(benchmarks):
    labels = benchmarks.keys()
    duration_sure = [x[0] for x in benchmarks.values()]
    duration_cv = [x[1] for x in benchmarks.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, duration_cv, width, label="5-fold CV")
    rects2 = ax.bar(x + width / 2, duration_sure, width, label="SURE")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Duration (s)")
    ax.set_title(
        "Hyperopt selection speed on MNE sample dataset", fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Plot samples data
    benchmarks_n_samples = joblib.load("data/simulated_samples_benchmarks.pkl")
    plot_benchmark_synthetic_data(
        benchmarks_n_samples,
        "samples",
        "Number of features: 20, number of tasks: 10",
    )

    # Plot features data
    benchmarks_n_features = joblib.load(
        "data/simulated_features_benchmarks.pkl"
    )
    plot_benchmark_synthetic_data(
        benchmarks_n_features,
        "features",
        "Number of samples: 10, number of tasks: 10",
    )

    # Plot tasks data
    benchmarks_n_tasks = joblib.load("data/simulated_tasks_benchmarks.pkl")
    plot_benchmark_synthetic_data(
        benchmarks_n_tasks,
        "tasks",
        "Number of samples: 10, number of features: 20",
    )

    # Plot true data
    benchmarks_meg = joblib.load("data/meg_benchmarks.pkl")
    plot_benchmark_meg_data(benchmarks_meg)
