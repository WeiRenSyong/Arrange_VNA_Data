"""
resonator_time_sweep_analysis.py
=================================
Time-sweep analysis of superconducting resonator S21 data.

Directory layout expected
-------------------------
<main_folder>/                             e.g. 15mK_Resonator_2_5p608GHz_Time_Dependent_S21
    time_sweep_summary.csv                 10-column CSV (header row)
                                           col D (index 3) → timestamp
                                           col F (index 5) → input power (e.g. "-20dBm")
    -20dBm/
        0.csv, 1.csv, 2.csv, ...           3-column S21 data (header row)
    -40dBm/
        0.csv, 1.csv, 2.csv, ...
    ...

Usage
-----
    python resonator_time_sweep_analysis.py <path_to_main_folder>

Or import and call `run_analysis(main_folder_path)` from another script.

Dependencies
------------
    pip install pandas numpy matplotlib scipy scresonators-fit
"""

# ── standard library ──────────────────────────────────────────────────────────
import argparse
import logging
import sys
import warnings
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import periodogram

# scresonators  (pip install scresonators-fit)
try:
    import fit_resonator.resonator as scres
    import fit_resonator.Sdata as fsd
    SCRESONATORS_AVAILABLE = True
except ImportError:
    SCRESONATORS_AVAILABLE = False
    warnings.warn(
        "scresonators-fit is not installed.  "
        "Install with:  pip install scresonators-fit\n"
        "Falling back to a self-contained DCM implementation.",
        stacklevel=2,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants & colour palette ────────────────────────────────────────────────
SUMMARY_COL_TIMESTAMP = 3   # column D  (0-indexed)
SUMMARY_COL_POWER     = 5   # column F  (0-indexed)

PARAM_LABELS = {
    "Qi":  r"$Q_i$",
    "Qc":  r"$Q_c$",
    "phi": r"$\phi$ (rad)",
    "fc":  r"$f_c$ (GHz)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DIRECTORY PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_main_folder(main_folder: Path) -> dict[str, list]:
    """
    Read time_sweep_summary.csv and return a dict mapping
      power_label → sorted list of (timestamp, csv_path) tuples.

    csv_path is resolved to the matching sequential file
    (0.csv, 1.csv, …) inside the power sub-folder.
    """
    summary_path = main_folder / "time_sweep_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    df = pd.read_csv(summary_path, header=0)

    # Accept either positional columns or named columns
    if df.shape[1] < max(SUMMARY_COL_TIMESTAMP, SUMMARY_COL_POWER) + 1:
        raise ValueError(
            f"time_sweep_summary.csv has only {df.shape[1]} columns; "
            f"expected at least {max(SUMMARY_COL_TIMESTAMP, SUMMARY_COL_POWER) + 1}."
        )

    timestamps_raw = df.iloc[:, SUMMARY_COL_TIMESTAMP]
    powers_raw     = df.iloc[:, SUMMARY_COL_POWER]

    # Parse timestamps – try numeric (Unix seconds) then string
    try:
        timestamps = pd.to_datetime(timestamps_raw, unit="s", utc=True)
    except Exception:
        timestamps = pd.to_datetime(timestamps_raw, infer_datetime_format=True, utc=True)

    # Normalise power labels: strip whitespace, ensure consistent format
    powers = powers_raw.astype(str).str.strip()

    # Group by power, sort chronologically, map to sequential CSV files
    power_map: dict[str, list] = {}
    for pwr in powers.unique():
        mask   = powers == pwr
        ts_grp = timestamps[mask].sort_values()
        pwr_dir = main_folder / pwr

        if not pwr_dir.is_dir():
            log.warning("Power subfolder not found, skipping: %s", pwr_dir)
            continue

        entries = []
        for idx, (_, ts) in enumerate(ts_grp.items()):
            csv_path = pwr_dir / f"{idx}.csv"
            if csv_path.exists():
                entries.append((ts, csv_path))
            else:
                log.warning("Expected file missing: %s", csv_path)

        if entries:
            power_map[pwr] = entries

    if not power_map:
        raise RuntimeError("No valid power/data pairings were found.")

    log.info("Found %d power levels: %s", len(power_map), sorted(power_map.keys()))
    return power_map


# ─────────────────────────────────────────────────────────────────────────────
# 2.  S21 LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_s21_csv(csv_path: Path) -> np.ndarray:
    """
    Read a 3-column S21 CSV (header row) and return an (N, 3) float array
    with columns [frequency_Hz, real_S21, imag_S21].
    """
    df = pd.read_csv(csv_path, header=0)
    if df.shape[1] < 3:
        raise ValueError(f"{csv_path}: expected 3 columns, got {df.shape[1]}")
    return df.iloc[:, :3].astype(float).values


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DCM FITTING
# ─────────────────────────────────────────────────────────────────────────────

# ── 3a.  scresonators wrapper ─────────────────────────────────────────────────

def _fit_dcm_scresonators(data: np.ndarray) -> dict | None:
    """
    Use the scresonators-fit library to perform a DCM fit.

    Returns a dict with keys Qi, Qi_err, Qc, Qc_err, phi, phi_err, fc, fc_err
    or None on failure.
    """
    try:
        method = scres.FitMethod(
            "DCM",
            MC_iteration=5,
            MC_rounds=500,
            MC_fix=["w1"],
            manual_init=None,
            MC_step_const=0.3,
        )
        result = fsd.fit(
            "tmp_fit",
            method,
            normalize=10,
            data_array=data,
            plot=False,
            verbose=False,
        )
        # result is a dict-like object; key names depend on scresonators version
        # Common keys: Qi, Qc, w1 (=fc), phi, and their _err counterparts
        if result is None:
            return None

        # Normalise key names across scresonators versions
        def _get(d, *keys, default=np.nan):
            for k in keys:
                if k in d:
                    return float(d[k])
            return default

        fc_hz = _get(result, "w1", "fc", "f_c")  # in Hz
        return {
            "Qi":     _get(result, "Qi", "Q_i"),
            "Qi_err": _get(result, "Qi_err", "Q_i_err", "dQi"),
            "Qc":     _get(result, "Qc", "Q_c"),
            "Qc_err": _get(result, "Qc_err", "Q_c_err", "dQc"),
            "phi":    _get(result, "phi"),
            "phi_err":_get(result, "phi_err", "dphi"),
            "fc":     fc_hz / 1e9 if not np.isnan(fc_hz) else np.nan,
            "fc_err": _get(result, "w1_err", "fc_err", "dfc") / 1e9
                      if not np.isnan(_get(result, "w1_err", "fc_err", "dfc")) else np.nan,
        }
    except Exception as exc:
        log.debug("scresonators fit exception: %s", exc)
        return None


# ── 3b.  self-contained fallback DCM implementation ──────────────────────────

def _dcm_model(f: np.ndarray, Ql: float, Qc: float, fc: float, phi: float,
               a: float, alpha: float, tau: float) -> np.ndarray:
    """
    Hanger-mode DCM S21 model (complex):
        S21(f) = a * exp(i*alpha) * exp(-2πi*f*tau)
                 * [1 - (Ql/Qc)*exp(iφ) / (1 + 2i*Ql*(f-fc)/fc)]
    Returns interleaved [Re, Im] for scipy curve_fit.
    """
    denom = 1.0 + 2j * Ql * (f - fc) / fc
    s21 = a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * tau) * (
        1.0 - (Ql / Qc) * np.exp(1j * phi) / denom
    )
    return np.concatenate([s21.real, s21.imag])


def _fit_dcm_fallback(data: np.ndarray) -> dict | None:
    """
    Self-contained Levenberg-Marquardt DCM fit used when scresonators is absent.
    Returns same dict as _fit_dcm_scresonators or None on failure.
    """
    from scipy.optimize import curve_fit

    freq = data[:, 0]
    s21c = data[:, 1] + 1j * data[:, 2]

    # ── initial guesses ──────────────────────────────────────────────────────
    mag   = np.abs(s21c)
    fc0   = freq[np.argmin(mag)]
    dip   = 1.0 - mag.min() / mag.max()
    Ql0   = fc0 / (freq[-1] - freq[0]) * 3.0   # rough guess
    Qc0   = Ql0 / max(dip, 1e-3)
    a0    = np.median(mag)
    alpha0, tau0, phi0 = 0.0, 0.0, 0.0

    p0     = [Ql0, Qc0, fc0, phi0, a0, alpha0, tau0]
    bounds = (
        [1e2,  1e2,  freq[0],  -np.pi, 0,    -np.pi, -1e-7],
        [1e9,  1e9,  freq[-1],  np.pi, 10,    np.pi,  1e-7],
    )

    y_data = np.concatenate([s21c.real, s21c.imag])

    try:
        popt, pcov = curve_fit(
            lambda f, *p: _dcm_model(f, *p),
            freq, y_data,
            p0=p0, bounds=bounds,
            maxfev=20000,
        )
    except Exception as exc:
        log.debug("Fallback DCM fit failed: %s", exc)
        return None

    perr = np.sqrt(np.diag(pcov))
    Ql, Qc, fc, phi, *_ = popt
    eQl, eQc, efc, ephi, *_ = perr

    # Qi from 1/Ql = 1/Qi + Re(1/Qc·exp(iφ))
    inv_Qi = 1.0 / Ql - np.cos(phi) / Qc
    if inv_Qi <= 0:
        return None
    Qi = 1.0 / inv_Qi
    # Error propagation (first-order)
    dInvQi_dQl  =  1.0 / Ql**2
    dInvQi_dQc  =  np.cos(phi) / Qc**2
    dInvQi_dphi =  np.sin(phi) / Qc
    Qi_err = Qi**2 * np.sqrt(
        (dInvQi_dQl * eQl)**2 +
        (dInvQi_dQc * eQc)**2 +
        (dInvQi_dphi * ephi)**2
    )

    return {
        "Qi":      Qi,
        "Qi_err":  Qi_err,
        "Qc":      Qc,
        "Qc_err":  eQc,
        "phi":     phi,
        "phi_err": ephi,
        "fc":      fc / 1e9,
        "fc_err":  efc / 1e9,
    }


# ── 3c.  public fit dispatcher ────────────────────────────────────────────────

def fit_s21(data: np.ndarray) -> dict | None:
    """
    Fit S21 data using DCM.  Prefers scresonators; falls back to built-in.
    Returns parameter dict or None.
    """
    if SCRESONATORS_AVAILABLE:
        result = _fit_dcm_scresonators(data)
        if result is not None:
            return result
        log.debug("scresonators returned None; trying fallback.")

    return _fit_dcm_fallback(data)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def process_power_map(
    power_map: dict[str, list],
) -> dict[str, pd.DataFrame]:
    """
    Iterate over every (timestamp, csv_path) pair, fit each file, and
    collect results into per-power DataFrames.

    Returns
    -------
    results : dict mapping power_label → DataFrame with columns
              [timestamp, Qi, Qi_err, Qc, Qc_err, phi, phi_err, fc, fc_err]
    """
    results: dict[str, pd.DataFrame] = {}

    for power, entries in sorted(power_map.items()):
        log.info("Processing power level %s  (%d files)", power, len(entries))
        rows = []
        for ts, csv_path in entries:
            try:
                data   = load_s21_csv(csv_path)
                params = fit_s21(data)
            except Exception as exc:
                log.warning("Skipping %s — load/fit error: %s", csv_path.name, exc)
                continue

            if params is None:
                log.warning("Skipping %s — fit returned no result.", csv_path.name)
                continue

            rows.append({"timestamp": ts, **params})

        if rows:
            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            results[power] = df
            log.info("  → %d successful fits", len(df))
        else:
            log.warning("  → No successful fits for power %s", power)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _power_colours(powers: list[str]) -> dict[str, tuple]:
    """Assign a perceptually distinct colour to each power label."""
    cmap   = cm.get_cmap("plasma", max(len(powers), 2))
    sorted_powers = sorted(powers)
    return {p: cmap(i / max(len(sorted_powers) - 1, 1))
            for i, p in enumerate(sorted_powers)}


def _time_axis(df: pd.DataFrame) -> np.ndarray:
    """Return elapsed time in seconds (float) relative to first point."""
    t0 = df["timestamp"].iloc[0]
    return (df["timestamp"] - t0).dt.total_seconds().values


# ── 5a.  Time-sweep trend plots ───────────────────────────────────────────────

def plot_time_sweeps(results: dict[str, pd.DataFrame],
                    title_prefix: str = "") -> plt.Figure:
    """
    Four-panel figure: Qi, Qc, φ, fc vs elapsed time.
    Each power level is a coloured line; uncertainty shown as shaded band.
    """
    params   = ["Qi", "Qc", "phi", "fc"]
    colours  = _power_colours(list(results.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    for ax, param in zip(axes, params):
        label_str = PARAM_LABELS[param]
        for power, df in sorted(results.items()):
            if param not in df.columns:
                continue
            t   = _time_axis(df)
            val = df[param].values
            err = df[f"{param}_err"].values

            c = colours[power]
            ax.plot(t, val, color=c, linewidth=1.5, label=power)
            ax.fill_between(t, val - err, val + err,
                            color=c, alpha=0.20, linewidth=0)

        ax.set_xlabel("Elapsed time (s)", fontsize=11)
        ax.set_ylabel(label_str, fontsize=12)
        ax.set_title(label_str, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Shared legend on first axes
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, title="Input power",
                       fontsize=9, title_fontsize=9,
                       loc="best", framealpha=0.7)

    fig.suptitle(
        f"{title_prefix}  — Time-Sweep Resonator Parameters",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return fig


# ── 5b.  PSD plots ────────────────────────────────────────────────────────────

def _compute_psd(values: np.ndarray,
                 timestamps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided PSD using scipy.signal.periodogram.
    Assumes approximately uniform sampling; uses median dt as sample period.

    Returns (frequencies_Hz, psd).
    """
    dt_sec = np.median(np.diff(
        timestamps.astype(np.int64).values / 1e9  # nanoseconds → seconds
    ))
    if dt_sec <= 0:
        dt_sec = 1.0
    fs = 1.0 / dt_sec

    # Detrend by subtracting mean to remove DC offset
    v = values - np.mean(values)
    freqs, psd = periodogram(v, fs=fs, window="hann", scaling="density")
    return freqs, psd


def plot_psd(results: dict[str, pd.DataFrame],
             title_prefix: str = "") -> plt.Figure:
    """
    Two-panel PSD figure: Qi noise and fc noise, overlapping power levels.
    """
    colours = _power_colours(list(results.keys()))

    fig, (ax_qi, ax_fc) = plt.subplots(1, 2, figsize=(14, 5))

    for power, df in sorted(results.items()):
        c = colours[power]

        if "Qi" in df.columns and len(df) > 4:
            f, p = _compute_psd(df["Qi"].values, df["timestamp"])
            ax_qi.loglog(f[1:], p[1:], color=c, linewidth=1.4, label=power)

        if "fc" in df.columns and len(df) > 4:
            # fc in GHz → convert to Hz fluctuations
            fc_hz = df["fc"].values * 1e9
            f, p  = _compute_psd(fc_hz, df["timestamp"])
            ax_fc.loglog(f[1:], p[1:], color=c, linewidth=1.4, label=power)

    for ax, title, ylabel in [
        (ax_qi, r"PSD of $Q_i$",        r"$S_{Q_i}$ [1/Hz]"),
        (ax_fc, r"PSD of $f_c$",        r"$S_{f_c}$ [Hz$^2$/Hz]"),
    ]:
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Input power",
                      fontsize=9, title_fontsize=9, framealpha=0.7)

    fig.suptitle(
        f"{title_prefix}  — Power Spectral Density of Noise",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write one CSV per power level to output_dir/fitted_params/."""
    out = output_dir / "fitted_params"
    out.mkdir(exist_ok=True)
    for power, df in results.items():
        safe = power.replace("/", "_").replace("\\", "_")
        path = out / f"{safe}.csv"
        df.to_csv(path, index=False)
        log.info("Saved fitted parameters → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(main_folder: str | Path,
                 save_plots: bool = True,
                 show_plots: bool = True) -> dict[str, pd.DataFrame]:
    """
    Full pipeline: parse → fit → plot → save.

    Parameters
    ----------
    main_folder : path to the top-level resonator folder
    save_plots  : write PNG files next to the data
    show_plots  : call plt.show() interactively

    Returns
    -------
    results : dict of power_label → DataFrame with fitted parameters
    """
    main_folder = Path(main_folder).resolve()
    if not main_folder.is_dir():
        raise FileNotFoundError(f"Main folder not found: {main_folder}")

    title = main_folder.name
    log.info("Starting analysis of: %s", title)

    # ── parse ─────────────────────────────────────────────────────────────────
    power_map = parse_main_folder(main_folder)

    # ── fit ───────────────────────────────────────────────────────────────────
    results = process_power_map(power_map)
    if not results:
        log.error("No fitting results produced.  Check data files and fit settings.")
        return results

    # ── save CSVs ─────────────────────────────────────────────────────────────
    save_results(results, main_folder)

    # ── time-sweep plot ───────────────────────────────────────────────────────
    fig_ts = plot_time_sweeps(results, title_prefix=title)
    if save_plots:
        p = main_folder / "time_sweep_parameters.png"
        fig_ts.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved time-sweep plot → %s", p)

    # ── PSD plot ──────────────────────────────────────────────────────────────
    fig_psd = plot_psd(results, title_prefix=title)
    if save_plots:
        p = main_folder / "psd_noise.png"
        fig_psd.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved PSD plot → %s", p)

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Time-sweep superconducting resonator analysis (DCM fitting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "main_folder",
        help="Path to the top-level resonator folder "
             "(e.g. 15mK_Resonator_2_5p608GHz_Time_Dependent_S21)",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Do not save PNG plots or CSV results to disk",
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Do not open interactive plot windows",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p


if __name__ == "__main__":
    args = _build_cli().parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    run_analysis(
        args.main_folder,
        save_plots=not args.no_save,
        show_plots=not args.no_show,
    )
