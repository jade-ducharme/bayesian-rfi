"""
Defines shared variables

TODO: Ensure you change these to your own directory paths!
"""
import os

import numpy as np

# Files and directories
ABS_DIR      = "/oscar/data/jpober/jmduchar/Research/BayesRFI/"  # top-level dir
DATA_IN_DIR  = "/users/jmduchar/data/shared/big_dataset/"  # raw visibilities input dir

HDF5_FILENAME             = "season_results.h5"
SOFT_LABEL_SAMPLES_SUBDIR = "soft_label_samples"

DATA_OUT_DIR  = os.path.join(ABS_DIR, "data/")        # pre-processed data output dir
RUN_DIR       = os.path.join(ABS_DIR, "out_aug16/")   # output dir for processed chains, etc.
UVFLAG_DIR    = os.path.join(DATA_OUT_DIR, "uvflag/")
CB_PATH       = os.path.join(os.path.dirname(__file__), "coarse_bands_1D.npy")
H5_PATH       = os.path.join(RUN_DIR, "processed/", HDF5_FILENAME)

# Experiment-specific variables
TV_SUBBANDS   = ['subTV', 'TV6', 'TV7', 'TV8', 'TV9']
POINTINGS     = [0, 1, 2, 3, 4]
POLARIZATIONS = ['xx', 'yy']
L             = 24
RANDOM_SEED   = 42
N_JOBS        = 50  # all combinations of (tv_subband,pointing,pol)

POL_TO_IDX = {
    'xx': 0,
    'yy': 1,
}

TV_BANDS = {
    'subTV': (0,       174e6),
    'TV6':   (174e6,   181e6),
    'TV7':   (181e6,   188e6),
    'TV8':   (188e6,   195e6),
    'TV9':   (195e6,   np.inf),
}

# Maps experiment-specific variables to a job_index
JOB_MAP = {
    idx: (L, tv, p, pol)
    for idx, (tv, p, pol) in enumerate(
        (tv, p, pol)
        for tv in TV_SUBBANDS
        for p in POINTINGS
        for pol in POLARIZATIONS
    )
}

# MWA GRIDNUM -> pointings east(+)/west(-) convention
POINTING_DICT  = {0: 0, 1: -1, 2: 1, 3: -2, 4: 2}
POINTING_ORDER = [3, 1, 0, 2, 4]
