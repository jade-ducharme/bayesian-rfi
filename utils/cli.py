"""
Shared command-line interface for the top-level pipeline scripts.
"""

import argparse
import os

_N_JOBS_DEST = "_n_jobs"

def build_parser(description, n_jobs, epilog=None):
    """
    Build an ArgumentParser carrying the CLI contract shared by every stage.

    Parameters
    ----------
    description : str
        One-line description of the stage, shown by `--help`.
    n_jobs : int
        Number of tasks in this stage's job grid (`len(JOB_MAP)`, `N_JOBS`, ...).
    epilog : str, optional
        Extra `--help` text, e.g. usage examples. Printed verbatim.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(**{_N_JOBS_DEST: n_jobs})

    # nargs="?" so `--n-jobs` can be used on its own
    parser.add_argument(
        "job_index",
        nargs="?",
        type=int,
        help=f"index of this task in the stage's job grid, 0 to {n_jobs - 1} "
             "(pass $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--n-jobs",
        action="store_true",
        dest="print_n_jobs",
        help="print the size of this stage's job grid (the #SBATCH --array bound) and exit",
    )
    return parser


def parse_args(parser, argv=None):
    """
    Parse and validate arguments from a parser built by `build_parser`.

    Returns
    -------
    argparse.Namespace
        With `job_index` guaranteed to be a valid index into the grid.
    """
    args = parser.parse_args(argv)
    n_jobs = parser.get_default(_N_JOBS_DEST)

    if args.print_n_jobs:
        print(n_jobs)
        raise SystemExit(0)

    if args.job_index is None:
        parser.error("job_index is required (pass $SLURM_ARRAY_TASK_ID)")

    if not 0 <= args.job_index < n_jobs:
        parser.error(f"job_index must be in [0, {n_jobs}), got {args.job_index}")

    return args


def add_run_dir(parser, default, help=None):
    """
    Add the `--run-dir` option shared by every stage that reads or writes a
    run's output tree (`{run_dir}/chains/`, `{run_dir}/processed/`, ...).

    The value may be absolute, or relative to the project root -- see
    `resolve_run_dir`.
    """
    parser.add_argument(
        "--run-dir",
        default=default,
        metavar="DIR",
        help=(help or "run output directory, absolute or relative to the "
                      "project root") + f" (default: {default})",
    )


def add_half(parser):
    """
    Add the `--half` option shared by the stages that support half-season
    splits (fit, postprocess, summarize_season). Omitted means the full season.
    """
    parser.add_argument(
        "--half",
        type=int,
        choices=(0, 1),
        default=None,
        help="restrict to half the season (0 or 1); omit for a full-season run",
    )


def resolve_run_dir(run_dir, abs_dir):
    """
    Resolve a `--run-dir` value to an absolute path.

    Absolute paths are used as given; anything else is interpreted relative to
    the project root
    """
    if os.path.isabs(run_dir):
        return run_dir
    return os.path.join(abs_dir, run_dir)


def print_config(title, **fields):
    """
    Print the banner every stage opens with: a title, then one `key : value`
    line per field, aligned. Field order is the order the keyword arguments
    were passed in.
    """
    print("\n----------------------")
    print(title.upper())
    print("----------------------\n")
    width = max((len(k) for k in fields), default=0)
    for key, value in fields.items():
        print(f"{key.ljust(width)} : {value}")
    print()
