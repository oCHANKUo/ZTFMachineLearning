import pandas as pd
import numpy as np
import pickle
import json

def diagnose_anomalies():
    """
    Run this and paste the output to show me what's going on.
    """
    with open("ztf_lcs_all.pkl", "rb") as f:
        all_lcs = pickle.load(f)

    with open("top_anomalies.json", "r") as f:
        anomalies = json.load(f)

    print("=" * 70)
    print("TOP 10 ANOMALIES - DATA QUALITY CHECK")
    print("=" * 70)

    for anom in anomalies[:10]:
        ztf_id = anom["ztf_id"]
        score = anom.get("raw_score", 0)
        primary = anom.get("primary_feature", "N/A")

        if ztf_id not in all_lcs:
            print(f"\n{ztf_id}: NOT FOUND in pickle")
            continue

        df = all_lcs[ztf_id]
        mag_col = "magpsf" if "magpsf" in df.columns else "magpsf_corr"

        n = len(df)
        n_g = len(df[df["fid"] == 1])
        n_r = len(df[df["fid"] == 2])

        mjds = df["mjd"].sort_values().values
        span = mjds[-1] - mjds[0] if len(mjds) > 1 else 0
        gaps = np.diff(mjds) if len(mjds) > 1 else []
        max_gap = np.max(gaps) if len(gaps) > 0 else 0

        mags = df[mag_col].values
        mag_range = np.max(mags) - np.min(mags)

        # Flag issues
        issues = []
        if n_g < 5: issues.append(f"g-sparse({n_g})")
        if n_r < 5: issues.append(f"r-sparse({n_r})")
        if max_gap > 50: issues.append(f"big-gap({max_gap:.0f}d)")
        if span < 5: issues.append(f"short({span:.1f}d)")
        if mag_range < 0.3: issues.append(f"flat(Δ{mag_range:.2f})")
        if n < 10: issues.append(f"few-pts({n})")

        issue_str = ", ".join(issues) if issues else "OK"

        print(f"\n{ztf_id} | Score:{score:.3f} | Key:{primary}")
        print(f"  n={n} (g={n_g}, r={n_r}) | span={span:.1f}d | max_gap={max_gap:.1f}d | Δmag={mag_range:.2f}")
        print(f"  Issues: {issue_str}")

        # Show a few data points
        print(f"  Data sample:")
        cols = ["mjd", mag_col, "fid"]
        if "magerr" in df.columns:
            cols.append("magerr")
        print(df[cols].head(3).to_string(index=False))

if __name__ == "__main__":
    diagnose_anomalies()