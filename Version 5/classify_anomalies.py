import json
import pandas as pd
import numpy as np
import pickle

def classify_anomalies():
    """
    Classify each anomaly as real, data artifact, or borderline.
    Run this after anomaly detection to decide what to keep.
    """

    # Load anomaly results
    with open("top_anomalies_strict.json", "r") as f:
        anomalies = json.load(f)

    # Load features
    df_features = pd.read_csv("ztf_features_strict.csv")
    df_features = df_features.set_index("ztf_id")

    # Load light curves
    with open("ztf_lcs_all.pkl", "rb") as f:
        all_lcs = pickle.load(f)

    print("=" * 70)
    print("ANOMALY CLASSIFICATION")
    print("=" * 70)

    for anom in anomalies[:10]:
        ztf_id = anom["ztf_id"]
        score = anom["raw_score"]
        primary = anom["primary_feature"]

        # Get feature row
        if ztf_id not in df_features.index:
            print(f"\n{ztf_id}: NOT IN FEATURES")
            continue

        row = df_features.loc[ztf_id]

        # Data quality metrics
        n_total = row.get("dq_total_points", 0)
        n_g = row.get("dq_g_frac", 0) * n_total
        n_r = row.get("dq_r_frac", 0) * n_total
        max_gap = row.get("dq_max_gap", 0)
        span = row.get("dq_span", 0)
        frac_imp = row.get("dq_frac_imputed", 0)

        # Light curve check
        if ztf_id in all_lcs:
            df_lc = all_lcs[ztf_id]
            mag_col = "magpsf" if "magpsf" in df_lc.columns else "magpsf_corr"
            mags = df_lc[mag_col].values
            mag_range = np.max(mags) - np.min(mags)

            # Check peak position
            peak_idx = np.argmin(mags)
            n_pre = peak_idx
            n_post = len(mags) - peak_idx - 1
        else:
            mag_range = 0
            n_pre = n_post = 0

        # CLASSIFICATION RULES
        issues = []

        if max_gap > 80:
            issues.append("large_gap")
        if n_g < 10 or n_r < 10:
            issues.append("sparse_band")
        if frac_imp > 0.05:
            issues.append("heavily_imputed")
        if n_pre < 5 or n_post < 5:
            issues.append("incomplete_peak")
        if mag_range < 1.0:
            issues.append("flat_curve")

        # Decision
        if len(issues) >= 3:
            verdict = "DATA ARTIFACT"
            confidence = "HIGH"
        elif len(issues) >= 1:
            verdict = "BORDERLINE"
            confidence = "MEDIUM"
        else:
            verdict = "REAL ASTROPHYSICAL"
            confidence = "HIGH"

        print(f"\n{ztf_id} | Score:{score:.3f} | Key:{primary}")
        print(f"  n={n_total:.0f} (g={n_g:.0f}, r={n_r:.0f}) | span={span:.0f}d | max_gap={max_gap:.0f}d | Δmag={mag_range:.2f}")
        print(f"  Pre-peak:{n_pre} | Post-peak:{n_post} | Imputed:{frac_imp:.1%}")
        print(f"  Issues: {', '.join(issues) if issues else 'NONE'}")
        print(f"  VERDICT: {verdict} (confidence: {confidence})")

    print("\n" + "=" * 70)
    print("GUIDELINES:")
    print("  DATA ARTIFACT → Remove from results, fix data quality")
    print("  BORDERLINE → Keep but flag, verify with visual inspection")
    print("  REAL ASTROPHYSICAL → These are your true anomalies!")
    print("=" * 70)

if __name__ == "__main__":
    classify_anomalies()