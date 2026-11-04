import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

import glob
import warnings

import cli
import hdf5_utils
import numpy as np
from pyuvdata import UVFlag

warnings.filterwarnings("ignore")

from config import (
    ABS_DIR,
    H5_PATH,
    N_JOBS,
    POL_TO_IDX,
    POLARIZATIONS,
    TV_BANDS,
    TV_SUBBANDS,
    UVFLAG_DIR,
)


def get_freq_mask(uvf, tv_subband):
    """Return boolean mask over freq_array for the given TV subband."""
    lo, hi = TV_BANDS[tv_subband]
    return (uvf.freq_array > lo) & (uvf.freq_array < hi)


def build_label_lookup(df):
    """
    Build a fast time -> {tv_subband: {pol: (soft_label, soft_label_std)}}
    lookup from the season-results dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'time_jd', 'tv_subband', 'pol', 'soft_label',
        'soft_label_std' (a subset of the flat dataframe from
        hdf5_utils.read_season_results).

    Returns
    -------
    lookup : dict
        Maps round(time_jd, TIME_DECIMALS) -> {tv_subband: {pol: (soft_label, soft_label_std)}}.
    """
    lookup = {}
    for row in df.itertuples(index=False):
        key = round(row.time_jd, 7)
        lookup.setdefault(key, {}).setdefault(row.tv_subband, {})[row.pol] = (
            row.soft_label, row.soft_label_std
        )
    return lookup


def build_rejected_lookup(rejected_obsids):
    """
    Invert hdf5_utils.read_rejected_obsids's per-combo obsid lists into a
    fast obsid -> {tv_subband, ...} lookup for propagate_flags_for_file.

    Parameters
    ----------
    rejected_obsids : dict
        (pointing, tv_subband) -> list of obsid strings, from
        `hdf5_utils.read_rejected_obsids`.

    Returns
    -------
    lookup : dict
        obsid -> set of tv_subband strings rejected for that obsid's night.
        An obsid belongs to exactly one pointing, so pointing isn't needed
        as part of the key.
    """
    lookup = {}
    for (_pointing, tv_subband), obsids in rejected_obsids.items():
        for obsid in obsids:
            lookup.setdefault(obsid, set()).add(tv_subband)
    return lookup


def propagate_flags_for_file(uvfile, lookup, rejected_lookup):
    """
    EXPERIMENTAL
    
    Propagate season-results RFI classifications into one UVFlag file, in
    place, and write the result back to disk in "metric" mode.

    Writes the continuous soft label (posterior mean RFI responsibility
    p_rfi, in [0, 1]) into `metric_array`, and its Bayesian error bar
    (posterior std of p_rfi) into `weights_array` -- pyuvdata's UVFlag schema
    has no dedicated uncertainty field, but `weights_array` is the only other
    float array of matching shape available in metric mode (see
    https://pyuvdata.readthedocs.io/en/latest/uvflag.html). This is a
    repurposing, not an established pyuvdata convention, so it's documented
    here and in the object's `history`.

    Parameters
    ----------
    uvfile : str
        Path to a UVFlag `*_flags.h5` file (read AND overwritten in place).
    lookup : dict
        Time -> {tv_subband: {pol: (soft_label, soft_label_std)}} lookup,
        from build_label_lookup.
    rejected_lookup : dict
        obsid -> set of tv_subband strings rejected for that obsid's night,
        from build_rejected_lookup.

    Returns
    -------
    n_missing : int
        Number of timesteps with no matching season-results entry (excluding
        those explained by a rejected sub-band).
    """
    uvf = UVFlag()
    uvf.read(uvfile)
    uvf.to_metric()
    uvf.history += (
        " metric_array = posterior mean soft label (p_rfi); "
        "weights_array = posterior std of the soft label (its Bayesian error "
        "bar), except weights=0 for force-flagged rejected sub-bands, which "
        "means 'no data' rather than 'zero uncertainty'. "
        "See propagate.py."
    )

    obsid = re.sub(r'_flags\.h5$', '', os.path.basename(uvfile))
    rejected_subbands = rejected_lookup.get(obsid, set())

    # ---- Rejected sub-bands: treat as fully RFI, no posterior information ----
    for tv_subband in rejected_subbands:
        freq_mask = get_freq_mask(uvf, tv_subband)
        if freq_mask.any():
            uvf.metric_array[:, freq_mask, :]  = 1.0
            uvf.weights_array[:, freq_mask, :] = 0.0
    if rejected_subbands:
        print(f"  {os.path.basename(uvfile)}: force-flagged rejected sub-band(s) "
              f"{sorted(rejected_subbands)} (excluded from mixture-model fit)")

    n_missing = 0
    for tdx, t in enumerate(uvf.time_array):
        labels_at_t = lookup.get(round(float(t), 7))
        if labels_at_t is None:
            n_missing += 1
            continue

        sub_metric  = uvf.metric_array[tdx, :, :]
        sub_weights = uvf.weights_array[tdx, :, :]

        for tv_subband in TV_SUBBANDS:
            if tv_subband in rejected_subbands:
                continue  # already handled above

            subband_labels = labels_at_t.get(tv_subband)
            if subband_labels is None:
                continue

            freq_mask = get_freq_mask(uvf, tv_subband)
            if not freq_mask.any():
                continue

            pol_values = {}
            for pol in POLARIZATIONS:
                if pol not in subband_labels:
                    continue
                soft_label, soft_label_std = subband_labels[pol]
                pol_idx = POL_TO_IDX[pol]
                sub_metric[freq_mask, pol_idx]  = soft_label
                sub_weights[freq_mask, pol_idx] = soft_label_std
                pol_values[pol] = (soft_label, soft_label_std)

            # Cross-polarization axes (XY, YX): take the more RFI-like of
            # xx/yy (max soft_label), carrying its error bar along --
            # the continuous analogue of the old OR-of-hard-labels.
            if pol_values:
                cross_pol = max(pol_values, key=lambda p: pol_values[p][0])
                cross_label, cross_std = pol_values[cross_pol]
                sub_metric[freq_mask, 2]  = cross_label
                sub_weights[freq_mask, 2] = cross_std
                sub_metric[freq_mask, 3]  = cross_label
                sub_weights[freq_mask, 3] = cross_std

        uvf.metric_array[tdx, :, :]  = sub_metric
        uvf.weights_array[tdx, :, :] = sub_weights

    uvf.write(uvfile, clobber=True)
    return n_missing


def propagate_flags(job_index, n_jobs, uvflag_dir=UVFLAG_DIR, h5_path=H5_PATH):
    """
    Propagate season-results RFI classifications into UVFlag files.

    Parameters
    ----------
    job_index : int
        Index of this job's chunk, 0 <= job_index < n_jobs.
    n_jobs : int
        Total number of parallel jobs the file list is split across.
    uvflag_dir : str, optional
        Directory containing `*_flags.h5` UVFlag files.
    h5_path : str, optional
        Path to the shared season_results.h5 file.
    """
    all_uvf = sorted(glob.glob(os.path.join(uvflag_dir, "*_flags.h5")))
    if not all_uvf:
        raise FileNotFoundError(f"No UVFlag files found in {uvflag_dir}")

    my_files = np.array_split(all_uvf, n_jobs)[job_index].tolist()
    print(f"Job {job_index}/{n_jobs}: processing {len(my_files)} of {len(all_uvf)} UVFlag files")

    print("Loading season results and building time lookup...")
    df = hdf5_utils.read_season_results(h5_path)
    lookup = build_label_lookup(df[['time_jd', 'tv_subband', 'pol', 'soft_label', 'soft_label_std']])
    del df

    print("Loading rejected nights and building obsid lookup...")
    rejected_obsids = hdf5_utils.read_rejected_obsids(h5_path)
    rejected_lookup = build_rejected_lookup(rejected_obsids)

    for uvfile in my_files:
        n_missing = propagate_flags_for_file(uvfile, lookup, rejected_lookup)
        print(f"  {os.path.basename(uvfile)}: {n_missing} timestep(s) with no matching season-results entry")

    print(f"Done with job {job_index}/{n_jobs}")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    parser = cli.build_parser(
        description="Pipeline stage 5: write this task's chunk of UVFlag files with "
                    "the soft labels and error bars from a run's season results.",
        n_jobs=N_JOBS,
        epilog="Examples:\n"
               "  python propagate.py 0\n"
               "  python propagate.py 0 --run-dir out_chrono_split\n"
               "  python propagate.py --n-jobs",
    )
    cli.add_run_dir(parser, default="out",
                    help="run directory to read processed/season_results.h5 from")
    args = cli.parse_args(parser)

    run_dir = cli.resolve_run_dir(args.run_dir, ABS_DIR)
    h5_path = os.path.join(run_dir, "processed", hdf5_utils.HDF5_FILENAME)

    cli.print_config(
        "Propagate flags",
        job_index  = args.job_index,
        n_jobs     = N_JOBS,
        run_dir    = run_dir,
        h5_path    = h5_path,
        uvflag_dir = UVFLAG_DIR,
    )

    propagate_flags(args.job_index, N_JOBS, h5_path=h5_path)
