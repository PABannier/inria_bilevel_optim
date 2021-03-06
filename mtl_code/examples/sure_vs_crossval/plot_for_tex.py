import matplotlib as mpl
import joblib
from celer.plot_utils import configure_plt
from seaborn import color_palette

import numpy as np
import matplotlib.pyplot as plt

from mtl.utils_plot import _plot_legend_apart

save_fig = True
# save_fig = False

configure_plt()


fontsize = 18

mpl.rcParams["xtick.labelsize"] = fontsize
mpl.rcParams["ytick.labelsize"] = fontsize

corrs = [0.5, 0.7, 0.9, 0.99]
criteria = ["mse", "sure", "f1"]
criteria_names = {}
criteria_names["mse"] = "MSE"
criteria_names["sure"] = "SURE"
criteria_names["f1"] = "F1"

dict_colors = color_palette("colorblind")


def load_data_for_corr(corr):
    infile_reweighted = f"data/scores_reweighted_corr_{corr}.pkl"
    infile_lasso = f"data/scores_lasso_corr_{corr}.pkl"

    reweighted_scores = joblib.load(infile_reweighted)
    lasso_scores = joblib.load(infile_lasso)

    alphas = reweighted_scores["alpha"]

    cvs = [
        {
            "f1": lasso_scores["f1"],
            "mse": lasso_scores["mse"],
            "sure": lasso_scores["sure"],
        },
        {
            "f1": reweighted_scores["f1"],
            "mse": reweighted_scores["mse"],
            "sure": reweighted_scores["sure"],
        },
    ]

    return alphas, cvs


# plots criterion * correlation
fig, axarr = plt.subplots(
    len(criteria), len(corrs), sharex="col", sharey="row"
)


for idx_corr, corr in enumerate(corrs):
    alphas, cvs = load_data_for_corr(corr)
    for idx_crit, criterion in enumerate(criteria):
        # MSE
        for idx_estim, estimator in enumerate(["Lasso", "reweighted Lasso"]):
            if criterion == "sure":
                y = cvs[idx_estim][criterion]
            else:
                y = cvs[idx_estim][criterion].mean(axis=1)

            if criterion == "f1":
                idx_opt = np.argmax(y)
            else:
                idx_opt = np.argmin(y)

            axarr[idx_crit, idx_corr].semilogx(
                alphas / alphas[0],
                y,
                color=dict_colors[idx_estim],
                lw=1,
                label=estimator,
            )
            axarr[idx_crit, idx_corr].axvline(
                alphas[idx_opt] / alphas[0],
                linestyle="--",
                lw=2,
                color=dict_colors[idx_estim],
                label=r"%s - best $\lambda$" % estimator,
            )
        # axarr[idx_crit, idx_corr].tick_params(
        #     axis='both', which='minor', labelsize=40)

    axarr[2, idx_corr].set_xlabel(
        r"$\lambda / \lambda_{\mathrm{max}}$", fontsize=fontsize
    )
    axarr[0, idx_corr].set_title(f"Corr = {corr}", fontsize=fontsize)

for idx_crit, criterion in enumerate(criteria):
    axarr[idx_crit, 0].set_ylabel(criteria_names[criterion], fontsize=fontsize)

plt.tight_layout()


# handles, labels = fig.axes[-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))
# plt.subplots_adjust(right=0.65)
fig.show()
# plt.show(block=True)

OUT_PATH_1 = (
    f"../../../tex/article/srcimages/metric_comparison/sure_mse_f1_comparison"
)
OUT_PATH_2 = (
    f"../../../tex/article/srcimages/metric_comparison/sure_mse_f1_comparison"
)

if save_fig:
    fig.savefig(OUT_PATH_1 + ".pdf")
    fig.savefig(OUT_PATH_2 + ".svg")
    _plot_legend_apart(axarr[0, 0], OUT_PATH_1 + "_legend.pdf")
    print("Figure saved.")
