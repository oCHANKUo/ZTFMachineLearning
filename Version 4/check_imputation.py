import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def check_imputation_quality():
    """
    Check if imputed points look reasonable by comparing them to real data.
    """
    with open("ztf_lcs_imputed.pkl", "rb") as f:
        imputed_lcs = pickle.load(f)

    with open("ztf_lcs_all.pkl", "rb") as f:
        original_lcs = pickle.load(f)

    print("=" * 60)
    print("IMPUTATION QUALITY CHECK")
    print("=" * 60)

    # Find objects that got imputed
    imputed_objects = []
    for ztf_id, df in imputed_lcs.items():
        if "imputed" in df.columns and df["imputed"].sum() > 0:
            imputed_objects.append(ztf_id)

    print(f"\nTotal objects with imputed points: {len(imputed_objects)}")

    # Sample 5 objects for detailed inspection
    sample = np.random.choice(imputed_objects, min(5, len(imputed_objects)), replace=False)

    for ztf_id in sample:
        df_imp = imputed_lcs[ztf_id]
        df_orig = original_lcs[ztf_id]

        mag_col = "magpsf" if "magpsf" in df_imp.columns else "magpsf_corr"

        real = df_imp[df_imp["imputed"] == False]
        synth = df_imp[df_imp["imputed"] == True]

        print(f"\n--- {ztf_id} ---")
        print(f"  Real points: {len(real)} | Synthetic: {len(synth)}")

        if len(synth) > 0:
            # Check if synthetic points are within reasonable range of real data
            real_mag_min = real[mag_col].min()
            real_mag_max = real[mag_col].max()
            synth_mags = synth[mag_col].values

            within_range = np.sum((synth_mags >= real_mag_min - 1) & (synth_mags <= real_mag_max + 1))
            print(f"  Synthetic mags within real range ±1: {within_range}/{len(synth)}")

            # Check if synthetic points fall in gaps (not on top of real data)
            for _, srow in synth.iterrows():
                t = srow["mjd"]
                nearest_real = np.min(np.abs(real["mjd"] - t))
                print(f"  Synthetic at MJD {t:.1f}: nearest real point = {nearest_real:.1f} days away")

    # Overall statistics
    print(f"\n--- Overall Statistics ---")
    total_real = 0
    total_synth = 0
    for ztf_id, df in imputed_lcs.items():
        if "imputed" in df.columns:
            total_real += (df["imputed"] == False).sum()
            total_synth += (df["imputed"] == True).sum()

    print(f"Total real points: {total_real}")
    print(f"Total synthetic points: {total_synth}")
    print(f"Synthetic fraction: {100*total_synth/(total_real+total_synth):.2f}%")

    if total_synth > 0.5 * (total_real + total_synth):
        print("\nWARNING: >50% synthetic! This is too much. Use stricter criteria.")
    elif total_synth > 0.2 * (total_real + total_synth):
        print("\nCAUTION: >20% synthetic. Consider stricter criteria.")
    else:
        print("\nOK: <20% synthetic. This is reasonable.")


def plot_imputation_comparison(ztf_id, output_file=None):
    """
    Plot original vs imputed light curve for a specific object.
    """
    with open("ztf_lcs_imputed.pkl", "rb") as f:
        imputed_lcs = pickle.load(f)

    with open("ztf_lcs_all.pkl", "rb") as f:
        original_lcs = pickle.load(f)

    if ztf_id not in imputed_lcs:
        print(f"{ztf_id} not found")
        return

    df_imp = imputed_lcs[ztf_id]
    df_orig = original_lcs[ztf_id]

    mag_col = "magpsf" if "magpsf" in df_imp.columns else "magpsf_corr"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'g': '#1a9850', 'r': '#d73027'}
    band_names = {1: 'g', 2: 'r'}

    # Plot original
    ax = axes[0]
    for fid, band in band_names.items():
        b = df_orig[df_orig["fid"] == fid].sort_values("mjd")
        if len(b) > 0:
            ax.scatter(b["mjd"], b[mag_col], c=colors[band], label=band, s=30, alpha=0.8)
    ax.invert_yaxis()
    ax.set_title(f"{ztf_id} - ORIGINAL")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot imputed
    ax = axes[1]
    for fid, band in band_names.items():
        b = df_imp[df_imp["fid"] == fid].sort_values("mjd")
        real = b[b["imputed"] == False] if "imputed" in b.columns else b
        synth = b[b["imputed"] == True] if "imputed" in b.columns else pd.DataFrame()

        if len(real) > 0:
            ax.scatter(real["mjd"], real[mag_col], c=colors[band], label=f"{band} (real)", s=30, alpha=0.8)
        if len(synth) > 0:
            ax.scatter(synth["mjd"], synth[mag_col], c=colors[band], marker='x', s=60, linewidths=2, label=f"{band} (synth)")

    ax.invert_yaxis()
    ax.set_title(f"{ztf_id} - IMPUTED")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    check_imputation_quality()

    # Uncomment to plot specific objects:
    # plot_imputation_comparison("ZTF18acbwaxk", "imputation_check.png")