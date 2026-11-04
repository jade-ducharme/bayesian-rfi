"""
Figure-making for the pipeline's diagnostic plots.
"""

import os

import arviz as az
import corner
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.time import Time
from matplotlib.colors import Normalize
from scipy.stats import norm, skewcauchy

import mixture_model

plt.style.use('seaborn-v0_8-white')

C_MODEL = "#04D9FF"
C_SSINS = "lime"
C_BOTH  = "red"

RFI_STYLES = {
    "model_only": {'label': "RFI: model only", 'marker': "s", 'color': C_MODEL,
                       's': 18, 'zorder': 3, 'edge': True},
    "ssins_only": {'label': "RFI: SSINS only", 'marker': "D", 'color': C_SSINS,
                       's': 12, 'zorder': 4, 'edge': True},
    "both":       {'label': "RFI: both",       'marker': "*", 'color': C_BOTH,
                       's': 18, 'zorder': 5, 'edge': False},
}

N_SIGMA_BG        = 10
N_SIGMA_ERRORBARS = 5
ERRORBAR_STAR_SIZE = 45


def markersize(s):
    return np.sqrt(s)


def rfi_masks(model_mask, ssins_mask):
    """
    Split one night's flagged samples into the three RFI categories the
    overlays are colour-coded by.

    Parameters
    ----------
    model_mask : ndarray of bool, shape (T,)
        True where the model's hard label says RFI.
    ssins_mask : ndarray of bool, shape (T,)
        True where SSINS' match filter flagged the sample.

    Returns
    -------
    dict
        Category name -> boolean mask
    """
    model_mask = np.asarray(model_mask).astype(bool)
    ssins_mask = np.asarray(ssins_mask).astype(bool)

    return {
        "model_only":  model_mask & ~ssins_mask,
        "ssins_only": ~model_mask &  ssins_mask,
        "both":        model_mask &  ssins_mask,
    }


def _styles(star_size=None):
    """`RFI_STYLES`, optionally with the star resized for a given figure."""
    if star_size is None:
        return RFI_STYLES

    return {name: (dict(style, s=star_size) if name == "both" else style)
            for name, style in RFI_STYLES.items()}


def scatter_rfi(ax, x, y, masks, star_size=None, label=True):
    """Overlay the flagged samples on a data panel"""
    for name, style in _styles(star_size).items():
        mask = masks[name]
        ax.scatter(
            x[mask], y[mask],
            marker=style['marker'],
            s=style['s'],
            color=style['color'],
            zorder=style['zorder'],
            label=style['label'] if label else None,
            **({'edgecolors': "black", 'linewidths': 0.4} if style['edge'] else {}),
        )


def errorbar_rfi(ax, x, y, yerr, masks, star_size=None):
    """
    Overlay the flagged samples on an errorbar panel
    """
    for name, style in _styles(star_size).items():
        mask = masks[name]
        ax.errorbar(
            x[mask], y[mask], yerr=yerr[mask],
            fmt=style['marker'],
            markersize=markersize(style['s']),
            color=style['color'],
            ecolor=style['color'],
            elinewidth=0.8,
            zorder=style['zorder'],
            **({'markeredgecolor': "black", 'markeredgewidth': 0.4}
               if style['edge'] else {}),
        )


def _moments(arr, ndx, stop):
    arr = np.asarray(arr[:, ndx, :stop], dtype=np.float64)

    return arr.mean(axis=0), arr.std(axis=0)


def plot_convergence_metrics(idata_summary, savefig=False, savepath=None, show=False):
    plt.figure(figsize=(6,3))
    plt.hist(np.array(idata_summary['r_hat'],dtype=float),bins=50)
    plt.title("Histogram of r-hat values for all sampled parameters, across all chains")
    plt.xticks(rotation=45)
    plt.yscale("log")
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath,"rhat.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(6,3))
    plt.hist(np.array(idata_summary['ess_bulk'],dtype=float),bins=50)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Histogram of ess_bulk values for all sampled parameters")
    if savefig:
        plt.savefig(os.path.join(savepath,"ess.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_trace_corner(idata, savefig=False, savepath=None, show=False):
    """
    Trace plots and a corner plot of the three global mixture parameters.
    """
    az.plot_trace(idata, var_names=[
                    'loc_rfi', 
                    'scale_rfi', 
                    'w_clean',
                ],
              figure_kwargs={"figsize": (12, 3)})
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath,"trace_plot.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    posterior = az.extract(idata,
        var_names=["loc_rfi", "scale_rfi", "w_rfi"]
    ).to_dataframe()

    fig = plt.figure(figsize=(5, 5))

    corner.corner(
        posterior[["loc_rfi", "scale_rfi", "w_rfi"]].values,
        labels=[r"$\mu_\mathrm{RFI}$", r"$\sigma_\mathrm{RFI}$", r"$w_\mathrm{RFI}$"],
        show_titles=True,
        fig=fig,
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 12},
        labelpad=0.1,
    )

    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=11)

    if savefig:
        plt.savefig(os.path.join(savepath,"param_corner.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def night_dates(night_ids):
    """
    UTC datetimes for a list of night ids.
    """
    return np.array([Time(int(n) * 10_000, format='gps').utc.datetime
                     for n in night_ids])


def _posterior_mean_beta(idata):
    """
    Posterior mean of `beta`, as an (N_nights, L) array.
    """
    beta     = idata.posterior['beta']
    n_chains = beta.sizes['chain']
    n_draws  = beta.sizes['draw']

    total = np.zeros(beta.shape[2:], dtype=np.float64)
    for c in range(n_chains):
        total += np.asarray(beta.isel(chain=c).values, dtype=np.float64).sum(axis=0)

    return total / (n_chains * n_draws)


def plot_night(idata, z, data_dict, ndx, mu_bg, pointing, sigma_clean,
                savefig=False, savepath=None, show=False):
    """
    Three-panel figure for one night: data with the fitted background,
    background-subtracted residuals, and the posterior soft labels, 
    sharing a time axis in seconds.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted posterior; used only for the per-night r-hat in the title.
    z : array-like, shape (S, N, T)
        Per-draw residuals, from `postprocessing_utils.get_bg_residuals`.
    data_dict : dict
        Padded per-night data, with the label arrays stage 3 adds to it
        (`hard_labels`, `soft_labels`, `soft_labels_std`).
    ndx : int
        Night index into the fitted night list.
    mu_bg : array-like, shape (S, N, T)
        Per-draw Legendre background, from `get_bg_residuals`.
    pointing : int
        Pointing, for the title.
    sigma_clean : array-like, shape (N,)
        Per-night analytic thermal noise, for the title.
    """

    rhat_beta = az.rhat(idata, var_names=["beta"])["beta"].values
    rhat_per_night = rhat_beta.mean(axis=-1)

    stop  = int(data_dict['lengths'][ndx])
    times = np.asarray(data_dict['times'])[ndx, :stop]
    x     = (times - times[0]) * 24 * 3600

    y          = np.asarray(data_dict['y_pad'])[ndx, :stop]
    ssins_mask = np.asarray(data_dict['f_pad'])[ndx, :stop].astype(bool)
    model_mask = np.asarray(data_dict['hard_labels'])[ndx, :stop].astype(bool)

    bg_mean, bg_std = _moments(mu_bg, ndx, stop)
    z_mean,  z_std  = _moments(z,     ndx, stop)

    label_mean = np.asarray(data_dict['soft_labels'])[ndx, :stop]
    label_std  = np.asarray(data_dict['soft_labels_std'])[ndx, :stop]

    masks = rfi_masks(model_mask, ssins_mask)

    unstarred = ~masks['both']

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.5, 0.8]}
    )

    # --- data + n-sigma background band ---------------------------------
    # Drawn before the band so that "Clean" leads the legend.
    axes[0].scatter(x, y, label="Clean", s=5, color="black")

    axes[0].fill_between(x, bg_mean - N_SIGMA_BG * bg_std,
                            bg_mean + N_SIGMA_BG * bg_std,
                         color='grey', alpha=0.3,
                         label=fr"Background ({N_SIGMA_BG}$\sigma$)")
    axes[0].plot(x, bg_mean, color='grey', linewidth=1.5)

    scatter_rfi(axes[0], x, y, masks)
    axes[0].legend()
    axes[0].set_ylabel("SSINS (arb. units)")
    axes[0].set_title(
        f"Pointing {pointing}, night {ndx+1} | "
        f"mean r-hat={rhat_per_night[ndx]:.3f} | "
        f"noise std={sigma_clean[ndx]:.3f}"
    )

    # --- residuals, n-sigma error bars ----------------------------------
    axes[1].errorbar(
        x[unstarred], z_mean[unstarred],
        yerr=N_SIGMA_ERRORBARS * z_std[unstarred],
        fmt="o", markersize=3, color="grey",
        ecolor="grey", elinewidth=0.8,
        label=fr"Error bars: {N_SIGMA_ERRORBARS}$\sigma$" "\n" "(this and lower panel)",
    )
    errorbar_rfi(axes[1], x, z_mean, N_SIGMA_ERRORBARS * z_std, masks,
                 star_size=ERRORBAR_STAR_SIZE)
    axes[1].axhline(0, linestyle="--", linewidth=1, color='grey')
    axes[1].set_ylabel("Residuals")

    leg = axes[1].legend(loc="upper left", fontsize=10, frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor("none")

    # --- soft labels, n-sigma error bars --------------------------------
    axes[2].errorbar(
        x[unstarred], label_mean[unstarred],
        yerr=N_SIGMA_ERRORBARS * label_std[unstarred],
        fmt="o", markersize=3, color="black",
        ecolor="black", elinewidth=0.8,
    )
    errorbar_rfi(axes[2], x, label_mean, N_SIGMA_ERRORBARS * label_std, masks,
                 star_size=ERRORBAR_STAR_SIZE)
    axes[2].axhline(0.5, linestyle="--", linewidth=1, color='black')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["0 - clean", "1 - RFI"])
    axes[2].set_ylabel("Soft label")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    if savefig:
        visdir = os.path.join(savepath, "visualize/")
        os.makedirs(visdir, exist_ok=True)
        plt.savefig(os.path.join(visdir, f"visualize_night{ndx:02d}.png"),
                    dpi=300,bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_clean_rfi_histograms(z, data_dict, samples, sigma_clean, savepath=None, savefig=False, show=False):
    """
    Histograms of the residuals the model called clean and RFI, each with the
    component it was assigned to drawn over it

    Parameters
    ----------
    z : array-like, shape (S, N, T)
        Per-draw residuals.
    data_dict : dict
        Output of prepare_multiple_nights, plus stage 3's 'hard_labels'.
    samples : dict
        Posterior samples from `extract_samples` (needs loc_rfi, scale_rfi).
    sigma_clean : array-like, shape (N,)
        Per-night thermal noise standard deviation.
    savepath : str, optional
        Directory to save the figure in.
    savefig : bool, optional
        Whether to save the figure to disk.
    show : bool, optional
        Whether to call plt.show().
    """
    A_RFI       = 0.99
    N_SIGMA_PDF = 10
    # Draws to build the RFI band from.
    MAX_PDF_DRAWS = 2000

    # Flatten residuals and labels across nights
    z_mean = np.mean(z, axis=0)  # (N, T)

    clean_samples = []
    rfi_samples   = []

    for plot_night in range(len(data_dict['lengths'])):
        stop    = int(data_dict['lengths'][plot_night])
        avg_z   = np.asarray(z_mean[plot_night, :stop])
        hlabels = np.asarray(data_dict['hard_labels'][plot_night, :stop]).astype(bool)

        finite = np.isfinite(avg_z)
        clean_samples.extend(avg_z[finite & ~hlabels])
        rfi_samples.extend(avg_z[finite & hlabels])

    clean_samples = np.asarray(clean_samples)
    rfi_samples   = np.asarray(rfi_samples)

    loc_full   = np.asarray(samples['loc_rfi']).ravel()
    scale_full = np.asarray(samples['scale_rfi']).ravel()
    if loc_full.size > MAX_PDF_DRAWS:
        pick       = np.random.default_rng(0).choice(loc_full.size, MAX_PDF_DRAWS, replace=False)
        loc_full   = loc_full[pick]
        scale_full = scale_full[pick]

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5))

    # ---- clean residuals vs. the analytic thermal noise ------------------
    axes[0].hist(clean_samples, density=True, bins=40,
                 color='black',
                 histtype="step",
                 linewidth=2,
                )
    axes[0].set_xlabel("Residuals - samples marked as clean")

    x_pdf = np.linspace(clean_samples.min(), clean_samples.max(), 500)
    for s in np.asarray(sigma_clean):
        axes[0].plot(x_pdf, norm.pdf(x_pdf, loc=0, scale=s),
                     color=sns.color_palette()[2], alpha=0.1)
    axes[0].set_yscale("log")
    axes[0].set_ylim(5e-4, 1.5)
    axes[0].set_ylabel("Probability density")

    # ---- RFI residuals vs. the fitted SkewCauchy -------------------------
    axes[1].hist(rfi_samples, density=True, bins=55,
                 color='black',
                 histtype="step",
                 linewidth=2,
                )
    axes[1].set_xlabel("Residuals - samples marked as RFI")
    axes[1].set_yscale("log")

    x_pdf = np.linspace(rfi_samples.min(), rfi_samples.max(), 1000)

    # The pdf for every kept draw -> (n_draws, n_x)
    curves = skewcauchy.pdf(
        x_pdf[None, :],
        a=A_RFI,
        loc=loc_full[:, None],
        scale=scale_full[:, None],
    )

    log_c   = np.log(np.clip(curves, 1e-300, None))
    mu, sd  = log_c.mean(axis=0), log_c.std(axis=0)
    lo, hi  = np.exp(mu - N_SIGMA_PDF * sd), np.exp(mu + N_SIGMA_PDF * sd)
    med     = np.exp(mu)

    axes[1].fill_between(x_pdf, lo, hi, color=sns.color_palette()[2], alpha=0.3, linewidth=0)
    axes[1].plot(x_pdf, med, color=sns.color_palette()[2], linewidth=1.2)
    axes[1].set_ylim(1e-6, 0.4)

    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath, "residuals.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def compare_model_pdfs(samples, sigma_clean, savefig=False, savepath=None, show=False):
    x = np.linspace(-10, 10, 1000)

    plt.figure(figsize=(8,4))

    for i in range(1000):
        label = 'SkewCauchy random draw' if i==0 else None
        idx = np.random.choice(samples['loc_rfi'].size)
        loc_rfi   = samples['loc_rfi'][idx]
        scale_rfi = samples['scale_rfi'][idx]
        skew_cauchy = mixture_model.SkewCauchy(loc_rfi, scale_rfi, a_rfi=0.99)
        pdf = np.exp(np.array(skew_cauchy.log_prob(jnp.array(x))))
        plt.plot(x, pdf, linewidth=2, label=label,
                alpha=0.05, color=sns.color_palette()[2])

    for sc in sigma_clean:
            plt.plot(x, norm.pdf(x, loc=0, scale=sc), color=sns.color_palette()[0], linewidth=1, alpha=0.2)
        
    plt.legend()
    # plt.yscale('log')
    # plt.ylim(1e-2,2)
    if savefig:
        plt.savefig(os.path.join(savepath, "compare_clean_rfi_histograms.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
