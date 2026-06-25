import pandas as pd
import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.interpolate import interp1d

def impute_lightcurve(ztf_id, df, verbose=False):
    """
    Impute missing light curve data using Gaussian Process regression.
    This is scientifically sound for astronomical time series.

    Strategy:
    1. If a band is completely missing, predict it from the other band using color
    2. If there are gaps, use GP to interpolate
    3. If only 1-2 points in a band, use GP with strong priors
    """

    df = df.copy().sort_values("mjd")
    mag_col = "magpsf" if "magpsf" in df.columns else "magpsf_corr"

    if verbose:
        print(f"\n--- Imputing {ztf_id} ---")

    # Separate bands
    g_data = df[df["fid"] == 1].copy()
    r_data = df[df["fid"] == 2].copy()

    # Common time grid for interpolation
    t_min = df["mjd"].min()
    t_max = df["mjd"].max()

    # If total span is very short, just return original
    if t_max - t_min < 1:
        return df

    # Strategy based on what data we have
    g_count = len(g_data)
    r_count = len(r_data)

    # CASE 1: Both bands have enough data - just fill small gaps with GP
    if g_count >= 5 and r_count >= 5:
        if verbose:
            print("  Both bands sufficient. Filling gaps with GP.")

        for band_data, fid in [(g_data, 1), (r_data, 2)]:
            if len(band_data) < 3:
                continue

            X = band_data["mjd"].values.reshape(-1, 1)
            y = band_data[mag_col].values

            # GP kernel: smooth variation + noise
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 100)) + WhiteKernel(0.1, (1e-5, 1))

            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0.1)
            gp.fit(X, y)

            # Find gaps > 5 days and add interpolated points
            mjds = band_data["mjd"].sort_values().values
            gaps = np.diff(mjds)

            new_points = []
            for i, gap in enumerate(gaps):
                if gap > 5:  # Significant gap
                    n_insert = int(gap / 5)  # One point every ~5 days
                    for j in range(1, n_insert + 1):
                        t_new = mjds[i] + j * (gap / (n_insert + 1))
                        y_pred, y_std = gp.predict([[t_new]], return_std=True)

                        new_points.append({
                            "mjd": t_new,
                            mag_col: y_pred[0],
                            "fid": fid,
                            "magerr": max(y_std[0], 0.05),  # Uncertainty from GP
                            "imputed": True
                        })

            if new_points:
                df = pd.concat([df, pd.DataFrame(new_points)], ignore_index=True)

    # CASE 2: One band is very sparse (< 5 points), predict from other band
    elif (g_count >= 5 and r_count < 5) or (r_count >= 5 and g_count < 5):

        rich_band = g_data if g_count >= 5 else r_data
        poor_band = r_data if g_count >= 5 else g_data
        rich_fid = 1 if g_count >= 5 else 2
        poor_fid = 2 if g_count >= 5 else 1

        if verbose:
            print(f"  {['g','r'][poor_fid-1]} band sparse ({len(poor_band)} pts). Predicting from {['g','r'][rich_fid-1]} band.")

        # Estimate color from overlapping observations
        if len(poor_band) >= 2:
            # We have some color measurements
            colors = []
            for _, poor_row in poor_band.iterrows():
                t = poor_row["mjd"]
                # Find nearest rich band point
                rich_near = rich_band.iloc[(rich_band["mjd"] - t).abs().argsort()[:1]]
                if len(rich_near) > 0 and abs(rich_near["mjd"].values[0] - t) < 2:
                    color = rich_near[mag_col].values[0] - poor_row[mag_col]
                    colors.append(color)

            mean_color = np.median(colors) if colors else 0.5  # Typical g-r ~ 0.5
        else:
            mean_color = 0.5  # Default assumption

        # Generate synthetic points for poor band
        new_points = []
        for _, rich_row in rich_band.iterrows():
            t = rich_row["mjd"]
            # Check if poor band already has point near this time
            if len(poor_band) > 0:
                nearest_gap = np.min(np.abs(poor_band["mjd"] - t))
                if nearest_gap < 1:  # Already have data here
                    continue

            # Predict magnitude from color
            if poor_fid == 1:  # g = r + color
                mag_pred = rich_row[mag_col] + mean_color
            else:  # r = g - color
                mag_pred = rich_row[mag_col] - mean_color

            new_points.append({
                "mjd": t,
                mag_col: mag_pred,
                "fid": poor_fid,
                "magerr": 0.3,  # Higher uncertainty for imputed
                "imputed": True
            })

        if new_points:
            df = pd.concat([df, pd.DataFrame(new_points)], ignore_index=True)
            if verbose:
                print(f"  Added {len(new_points)} imputed points.")

    # CASE 3: Both bands sparse - can't do much, return original
    else:
        if verbose:
            print("  Both bands too sparse. Returning original.")

    # Sort and deduplicate
    df = df.sort_values(["mjd", "fid"]).reset_index(drop=True)

    # Mark original data
    if "imputed" not in df.columns:
        df["imputed"] = False

    return df


def run_imputation():
    """
    Run imputation on all light curves and re-extract features.
    """
    with open("ztf_lcs_all.pkl", "rb") as f:
        all_lcs = pickle.load(f)

    print(f"Imputing {len(all_lcs)} light curves...")

    imputed_lcs = {}
    imputation_stats = {"imputed": 0, "unchanged": 0, "failed": 0}

    for i, (ztf_id, df) in enumerate(all_lcs.items()):
        try:
            df_imputed = impute_lightcurve(ztf_id, df, verbose=False)

            n_new = df_imputed["imputed"].sum() if "imputed" in df_imputed.columns else 0

            if n_new > 0:
                imputation_stats["imputed"] += 1
            else:
                imputation_stats["unchanged"] += 1

            imputed_lcs[ztf_id] = df_imputed

        except Exception as e:
            imputation_stats["failed"] += 1
            imputed_lcs[ztf_id] = df  # Keep original on failure

        if i % 100 == 0:
            print(f"  Progress: {i}/{len(all_lcs)}")

    # Save imputed light curves
    with open("ztf_lcs_imputed.pkl", "wb") as f:
        pickle.dump(imputed_lcs, f)

    print(f"\nImputation complete:")
    print(f"  Imputed: {imputation_stats['imputed']}")
    print(f"  Unchanged: {imputation_stats['unchanged']}")
    print(f"  Failed: {imputation_stats['failed']}")
    print(f"\nSaved to ztf_lcs_imputed.pkl")

    return imputed_lcs


if __name__ == "__main__":
    run_imputation()