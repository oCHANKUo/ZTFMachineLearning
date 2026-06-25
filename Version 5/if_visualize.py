import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import pickle

def plot_anomaly_lightcurves(json_path="top_anomalies_strict.json", 
                              pkl_path="ztf_lcs_imputed.pkl",
                              output_file="anomaly_lightcurves.png",
                              top_n=10,
                              phase_fold=False):
    """
    Plot top anomaly light curves in a scientific publication-ready format.

    Parameters:
    -----------
    json_path : str
        Path to anomaly JSON file
    pkl_path : str
        Path to pickle file with light curves
    output_file : str
        Output PNG filename
    top_n : int
        Number of anomalies to plot
    phase_fold : bool
        If True, fold light curves to peak (day 0 = peak)
    """

    # Load data
    with open(json_path, "r") as f:
        anomalies = json.load(f)

    with open(pkl_path, "rb") as f:
        all_lcs = pickle.load(f)

    anomalies = anomalies[:top_n]
    n = len(anomalies)

    if n == 0:
        print("No anomalies found.")
        return

    # Layout
    ncols = 2
    nrows = (n + 1) // 2

    fig = plt.figure(figsize=(14, 3.5 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.4, wspace=0.25,
                  left=0.06, right=0.98, top=0.96, bottom=0.04)

    colors = {'g': '#1a9850', 'r': '#d73027'}  # Colorblind-friendly
    band_names = {1: 'g', 2: 'r'}

    for i, anom in enumerate(anomalies):
        ztf_id = anom["ztf_id"]
        label = anom.get("label", "Unknown")
        score = anom.get("raw_score", 0)
        primary_feat = anom.get("primary_feature", "N/A")
        deviation = anom.get("deviation", 0)

        ax = fig.add_subplot(gs[i // ncols, i % ncols])

        if ztf_id not in all_lcs:
            ax.text(0.5, 0.5, f"{ztf_id}\n(no data)", 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, style='italic', color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        df = all_lcs[ztf_id].copy()

        # Determine phase offset
        if phase_fold:
            mag_col = "magpsf" if "magpsf" in df.columns else "magpsf_corr"
            peak_mjd = df.loc[df[mag_col].idxmin(), "mjd"] if len(df) > 0 else df["mjd"].min()
            df["phase"] = df["mjd"] - peak_mjd
            x_col = "phase"
            xlabel = "Phase (days from peak)"
        else:
            t0 = df["mjd"].min()
            df["days"] = df["mjd"] - t0
            x_col = "days"
            xlabel = "Days since first detection"

        # Plot each band
        for fid, band_name in band_names.items():
            band_data = df[df["fid"] == fid].sort_values(x_col)

            if len(band_data) == 0:
                continue

            mag_col = "magpsf" if "magpsf" in band_data.columns else "magpsf_corr"
            x = band_data[x_col].values
            y = band_data[mag_col].values

            if "magerr" in band_data.columns:
                yerr = band_data["magerr"].values
            else:
                yerr = None

            ax.errorbar(x, y, yerr=yerr,
                       fmt='o', color=colors[band_name], 
                       markersize=3.5, capsize=1.5, elinewidth=0.8,
                       alpha=0.85, label=f'{band_name}')

            # Connect points with thin line
            ax.plot(x, y, '-', color=colors[band_name], alpha=0.3, linewidth=0.8)

        # Formatting
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Magnitude", fontsize=8)

        title = f"{ztf_id}  |  {label}"
        subtitle = f"IF Score: {score:.3f}  |  Dev: {deviation:.1f}σ  |  {primary_feat}"
        ax.set_title(title, fontsize=9, fontweight='bold', pad=2)
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes,
               fontsize=7, va='top', ha='left', color='#444444',
               style='italic')

        ax.legend(loc='lower right', fontsize=6.5, framealpha=0.9, 
                 edgecolor='gray', fancybox=False)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=7)

        # Light red background for anomalies
        ax.set_facecolor('#fff5f5')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#cc0000')

    fig.suptitle("Top Anomalies — Isolation Forest Detection", 
                fontsize=13, fontweight='bold', y=0.99)

    plt.savefig(output_file, dpi=250, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Saved: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_anomaly_lightcurves()