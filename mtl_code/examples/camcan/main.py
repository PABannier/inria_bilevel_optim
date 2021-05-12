from pathlib import Path
import os
from joblib import parallel_backend
from joblib import Parallel, delayed
import glob

from utils_solver import solve_inverse_problem, generate_report

DATA_PATH = os.path.abspath(
    "/storage/store2/work/rhochenb/Data/Cam-CAN/BIDS/derivatives/mne-study-template"
)

OUT_PATH = os.path.abspath(
    "/storage/store2/work/pbannier/inria_bilevel_optim/mtl_code/examples/camcan/reports"
)

N_JOBS = 1  # -1
INNER_MAX_NUM_THREADS = 1

LOOSE = 0  # 0 for fixed, 0.9 for free


def solve_for_patient(folder_name):
    print(f"Solving {folder_name}...")

    patient_path = DATA_PATH / folder_name
    stc, residual, evoked, noise_cov = solve_inverse_problem(
        folder_name, patient_path, LOOSE
    )

    out_report_path = OUT_PATH / f"{folder_name}.html"

    generate_report(
        folder_name, out_report_path, stc, evoked, residual, noise_cov
    )


if __name__ == "__main__":
    DATA_PATH = Path(DATA_PATH)
    patient_folders = glob.glob(DATA_PATH / "*")

    with parallel_backend("loky", inner_max_num_threads=INNER_MAX_NUM_THREADS):
        Parallel(N_JOBS)(
            delayed(solve_for_patient)(folder_name)
            for folder_name in patient_folders
        )
