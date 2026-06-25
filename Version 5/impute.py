import pandas as pd
import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def impute_lightcurve(ztf_id, df, verbose=False):
    """
    Conservative imputation: only fill LARGE gaps that actually affect features.

    Strategy:
    1. Only impute gaps > 30 days (not the normal 3-day ZTF cadence)
    2. Max 5 imputed points per object (don't over-synthesize)
    3. If one band has < 5 points, predict from other band using color
    4. Mark all imputed points clearly
    """

    df = df.copy().sort_values("mjd")
    mag_col = "magpsf" if "magpsf" in df.columns else "magpsf_corr"

    g_data = df[df["fid"] == 1].copy()
    r_data = df[df["fid"] == 2].copy()

    t_min = df["mjd"].min()
    t_max = df["mjd"].max()

    if t_max - t_min < 1:
        df["imputed"] = False
        return df

    g_count = len(g_data)
    r_count = len(r_data)

    total_imputed = 0
    MAX_IMPUTED = 5  # Hard limit per object
    GAP_THRESHOLD = 30  # Only fill gaps > 30 days (not normal 3-day cadence)

    # CASE 1: Both bands well-sampled - GP interpolate ONLY large gaps
    if g_count >= 5 and r_count >= 5:
        for band_data, fid in [(g_data, 1), (r_data, 2)]:
            if len(band_data) < 3 or total_imputed >= MAX_IMPUTED:
                continue

            X = band_data["mjd"].values.reshape(-1, 1)
            y = band_data[mag_col].values

            # Simpler kernel for stability
            kernel = RBF(20, (5, 100)) + WhiteKernel(0.1, (1e-5, 1))

            try:
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.2)
                gp.fit(X, y)

                mjds = band_data["mjd"].sort_values().values
                gaps = np.diff(mjds)

                new_points = []
                for j, gap in enumerate(gaps):
                    if gap > GAP_THRESHOLD and total_imputed < MAX_IMPUTED:
                        # Only add ONE point in the middle of large gap
                        t_new = mjds[j] + gap / 2
                        y_pred, y_std = gp.predict([[t_new]], return_std=True)

                        new_points.append({
                            "mjd": t_new,
                            mag_col: y_pred[0],
                            "fid": fid,
                            "magerr": max(y_std[0], 0.1),
                            "imputed": True
                        })
                        total_imputed += 1

                if new_points:
                    df = pd.concat([df, pd.DataFrame(new_points)], ignore_index=True)

            except Exception as e:
                pass

    # CASE 2: One band sparse - predict from other band
    elif (g_count >= 5 and r_count < 5) or (r_count >= 5 and g_count < 5):

        rich_band = g_data if g_count >= 5 else r_data
        poor_band = r_data if g_count >= 5 else g_data
        rich_fid = 1 if g_count >= 5 else 2
        poor_fid = 2 if g_count >= 5 else 1

        # Estimate color from overlapping observations
        colors = []
        for _, poor_row in poor_band.iterrows():
            t = poor_row["mjd"]
            rich_near = rich_band.iloc[(rich_band["mjd"] - t).abs().argsort()[:1]]
            if len(rich_near) > 0 and abs(rich_near["mjd"].values[0] - t) < 2:
                color = rich_near[mag_col].values[0] - poor_row[mag_col]
                colors.append(color)

        mean_color = np.median(colors) if colors else 0.5
        color_std = np.std(colors) if len(colors) > 1 else 0.3

        new_points = []
        for _, rich_row in rich_band.iterrows():
            if total_imputed >= MAX_IMPUTED:
                break

            t = rich_row["mjd"]
            if len(poor_band) > 0:
                nearest_gap = np.min(np.abs(poor_band["mjd"] - t))
                if nearest_gap < 1:
                    continue

            if poor_fid == 1:
                mag_pred = rich_row[mag_col] + mean_color
            else:
                mag_pred = rich_row[mag_col] - mean_color

            new_points.append({
                "mjd": t,
                mag_col: mag_pred,
                "fid": poor_fid,
                "magerr": max(color_std, 0.3),
                "imputed": True
            })
            total_imputed += 1

        if new_points:
            df = pd.concat([df, pd.DataFrame(new_points)], ignore_index=True)

    df = df.sort_values(["mjd", "fid"]).reset_index(drop=True)
    if "imputed" not in df.columns:
        df["imputed"] = False

    return df


def run_imputation():
    with open("ztf_lcs_all.pkl", "rb") as f:
        all_lcs = pickle.load(f)

    print(f"Imputing {len(all_lcs)} light curves...")
    print(f"Gap threshold: 30 days | Max imputed per object: 5")

    imputed_lcs = {}
    stats = {"imputed": 0, "unchanged": 0, "failed": 0}
    total_added = 0
    imputed_counts = []  # Track how many points were added per object

    for i, (ztf_id, df) in enumerate(all_lcs.items()):
        try:
            df_imputed = impute_lightcurve(ztf_id, df, verbose=False)
            n_new = df_imputed["imputed"].sum() if "imputed" in df_imputed.columns else 0
            total_added += n_new
            imputed_counts.append(n_new)

            if n_new > 0:
                stats["imputed"] += 1
            else:
                stats["unchanged"] += 1

            imputed_lcs[ztf_id] = df_imputed

        except Exception as e:
            stats["failed"] += 1
            imputed_lcs[ztf_id] = df

        if i % 100 == 0:
            print(f"  Progress: {i}/{len(all_lcs)}")

    with open("ztf_lcs_imputed.pkl", "wb") as f:
        pickle.dump(imputed_lcs, f)

    print(f"\nImputation complete:")
    print(f"  Imputed: {stats['imputed']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total synthetic points added: {total_added}")
    print(f"  Avg imputed per object: {np.mean(imputed_counts):.2f}")
    print(f"  Max imputed for single object: {max(imputed_counts)}")
    print(f"\nDistribution of imputed points per object:")
    for n in range(6):
        count = sum(1 for x in imputed_counts if x == n)
        print(f"  {n} points: {count} objects")
    print(f"\nSaved to ztf_lcs_imputed.pkl")

    return imputed_lcs


if __name__ == "__main__":
    run_imputation()