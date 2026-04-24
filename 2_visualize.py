import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

# ANOMALIES_IN    = "top_anomalies.json"
ANOMALIES_IN    = "ocsvm_anomalies.json"
LIGHTCURVES_PKL = "ztf_lcs_all.pkl"
OUTPUT_FILE     = "top_anomalies.png"   # set None to show interactively

BANDS = [
    {"fid": 1, "color": "#4CAF50", "label": "g-band"},
    {"fid": 2, "color": "#EF5350", "label": "r-band"},
    {"fid": 3, "color": "#FF9800", "label": "i-band"},
]


def _mag_col(df):
    for col in ("magpsf_corr", "magpsf"):
        if col in df.columns:
            return col
    raise KeyError("No recognised magnitude column found.")


def match_id(ztf_id, lcs_keys):
    """
    Try to find ztf_id in the light-curve dict despite case/whitespace mismatches.
    Returns the actual key if found, else None.
    """
    if ztf_id in lcs_keys:
        return ztf_id
    ztf_id_lower = ztf_id.strip().lower()
    for k in lcs_keys:
        if k.strip().lower() == ztf_id_lower:
            return k
    return None


def plot_lightcurve(ax, lc_df, ztf_id, meta):
    mag_col = _mag_col(lc_df)
    err_col = mag_col.replace("magpsf", "sigmapsf")
    if err_col == mag_col:
        err_col = None

    plotted_any = False
    peak_t = None
    peak_m = None

    for band in BANDS:
        sub = lc_df[lc_df["fid"] == band["fid"]].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("mjd")
        t = sub["mjd"].values
        m = sub[mag_col].values

        # Drop NaNs
        valid = np.isfinite(t) & np.isfinite(m)
        t, m = t[valid], m[valid]
        if len(t) < 2:
            continue

        # Remove 5σ outliers
        med, std = np.median(m), np.std(m)
        if std > 0:
            mask = np.abs(m - med) < 5 * std
            t, m = t[mask], m[mask]
        if len(t) < 2:
            continue

        t_rel = t - t.min()

        # Error bars — re-filter sub to align safely
        yerr = None
        if err_col and err_col in sub.columns:
            sub2 = sub[np.isfinite(sub["mjd"]) & np.isfinite(sub[mag_col])].copy()
            if std > 0:
                sub2 = sub2[np.abs(sub2[mag_col] - med) < 5 * std]
            if len(sub2) == len(t):
                yerr = np.clip(sub2[err_col].values, 0, 1.5)

        # Only show error bars when they're large enough to be meaningful
        # (> 0.01 mag). Tiny/zero errors just render as a distracting vertical
        # line through the point, so we suppress them.
        plot_yerr = None
        if yerr is not None:
            median_err = np.median(yerr)
            if median_err > 0.01:
                plot_yerr = yerr

        ax.errorbar(
            t_rel, m, yerr=plot_yerr,
            fmt="o", color=band["color"],
            markersize=4, linewidth=0,
            elinewidth=0.8, capsize=0,   # capsize=0 removes the horizontal end caps
            alpha=0.75, label=band["label"], zorder=3,
        )

        if len(m) >= 5:
            window = max(3, len(m) // 10)
            smooth = pd.Series(m).rolling(window, center=True, min_periods=1).median()
            ax.plot(t_rel, smooth, color=band["color"], linewidth=1.8, alpha=0.95, zorder=4)

        plotted_any = True

        if peak_m is None or m.min() < peak_m:
            peak_m = m.min()
            peak_t = t_rel[np.argmin(m)]

    if not plotted_any:
        ax.text(0.5, 0.5, "No photometry data\nfor this object",
                transform=ax.transAxes, ha="center", va="center",
                color="#aaa", fontsize=8, linespacing=1.8)
        return

    if peak_t is not None and peak_t > 0:
        ax.axvspan(0, peak_t, alpha=0.07, color="yellow", zorder=1)

    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=7, colors="#ccc")

    feat  = meta.get("primary_feature", "?")
    z     = meta.get("primary_z", 0)
    score = meta.get("raw_score", 0)
    label = meta.get("label", "unknown")

    ax.set_title(
        f"{ztf_id}   [{label}]\n"
        f"score {score:.3f}  |  {feat} ({z:+.1f}σ)",
        fontsize=8, pad=6, loc="left", color="#eeeeee",
    )
    ax.set_xlabel("Days since first observation", fontsize=7, color="#aaaaaa")
    ax.set_ylabel("Magnitude  (brighter ↑)", fontsize=7, color="#aaaaaa")

    legend = ax.legend(fontsize=6, loc="lower right", framealpha=0.5,
                       handlelength=1.2, markerscale=0.8,
                       labelcolor="#cccccc", facecolor="#222233",
                       edgecolor="#444455")
    legend.get_frame().set_linewidth(0.5)


def main():
    with open(ANOMALIES_IN) as f:
        anomalies = json.load(f)

    with open(LIGHTCURVES_PKL, "rb") as f:
        lcs = pickle.load(f)

    lcs_keys = set(lcs.keys())

    # Diagnose ID mismatches before plotting
    print(f"\nLight-curve dict: {len(lcs_keys)} objects.")
    print(f"Sample keys from pkl: {list(lcs_keys)[:5]}\n")

    ids      = [a["ztf_id"] for a in anomalies]
    meta_map = {a["ztf_id"]: a for a in anomalies}

    resolved = {}
    for ztf_id in ids:
        key = match_id(ztf_id, lcs_keys)
        resolved[ztf_id] = key
        if key is None:
            print(f"  WARNING: '{ztf_id}' not found in pkl (case-insensitive search also failed).")
        elif key != ztf_id:
            print(f"  NOTICE:  '{ztf_id}' matched as '{key}' (case/whitespace fix applied).")

    n      = len(ids)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    fig = plt.figure(
        figsize=(6.5 * n_cols, 4.5 * n_rows),
        facecolor="#0f0f0f",
    )
    fig.suptitle(
        f"ZTF Anomaly Detection  ·  Top {n} Objects",
        fontsize=13, color="white", fontweight="bold",
    )

    # Explicit margins — avoids tight_layout clipping the suptitle or panel titles
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.65,
        wspace=0.35,
        top=0.93,
        bottom=0.06,
        left=0.07,
        right=0.97,
    )

    for i, ztf_id in enumerate(ids):
        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555566")
            spine.set_linewidth(0.6)
        ax.tick_params(colors="#aaaaaa")

        lc_key = resolved[ztf_id]
        if lc_key is not None:
            plot_lightcurve(ax, lcs[lc_key], ztf_id, meta_map[ztf_id])
        else:
            ax.set_facecolor("#110011")
            ax.text(0.5, 0.58, ztf_id, transform=ax.transAxes,
                    ha="center", va="center", color="#ff8888",
                    fontsize=8, fontweight="bold")
            ax.text(0.5, 0.40,
                    "ID not found in ztf_lcs_all.pkl\n"
                    "Check for case or format mismatch\n"
                    "(see terminal output for details)",
                    transform=ax.transAxes,
                    ha="center", va="center", color="#888888",
                    fontsize=7, linespacing=1.7)

    for i in range(n, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        fig.add_subplot(gs[r, c]).set_visible(False)

    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nSaved → {OUTPUT_FILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()