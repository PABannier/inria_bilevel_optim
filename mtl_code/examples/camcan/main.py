from pathlib import Path
import os
from joblib import parallel_backend
from joblib import Parallel, delayed
import glob

from utils_solver import solve_inverse_problem, generate_report

from mne.viz import plot_sparse_source_estimates

DATA_PATH = Path(
    "../../../../../rhochenb/Data/Cam-CAN/BIDS/derivatives/mne-study-template"
)

OUT_PATH = Path("reports")

N_JOBS = 1  # -1
INNER_MAX_NUM_THREADS = 1

LOOSE = 0  # 0 for fixed, 0.9 for free


def solve_for_patient(folder_path):
    folder_name = folder_path.split("/")[-1]
    print(f"Solving #{folder_name}")

    patient_path = DATA_PATH / folder_name
    (
        stc,
        residual,
        evoked,
        noise_cov,
        subject_dir,
        forward,
    ) = solve_inverse_problem(folder_name, patient_path, LOOSE)

    # plot_sparse_source_estimates(
    #    forward["src"], stc, bgcolor=(1, 1, 1), opacity=0.1
    # )

    # out_report_path = OUT_PATH / f"{folder_name}.html"

    # generate_report(
    #     folder_name,
    #     out_report_path,
    #     stc,
    #     evoked,
    #     residual,
    #     noise_cov,
    #     subject_dir,
    # )


if __name__ == "__main__":
    patient_folders = glob.glob(str(DATA_PATH / "*"))
    patient_folders = [x for x in patient_folders if os.path.isdir(x)]

    with parallel_backend("loky", inner_max_num_threads=INNER_MAX_NUM_THREADS):
        Parallel(N_JOBS)(
            delayed(solve_for_patient)(folder_name)
            for folder_name in patient_folders
        )
