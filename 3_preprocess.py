import pandas as pd
import pickle
import numpy as np

def mag_to_flux(mag):
    return 10**(-0.4 * (mag - 25.0))

def extract_features(ztf_id, df, label, redshift):
    # 1. QUALITY FILTER: Only keep high-confidence detections
    if 'magerr' in df.columns:
        df = df[df['magerr'] < 0.2]
    
    # Check if empty after filtering
    if len(df) < 5: return None
    
    # Check for placeholder magnitudes (artifacts)
    mag_col = 'magpsf' if 'magpsf' in df.columns else 'magpsf_corr'
    if df[mag_col].max() > 30: return None

    # 2. TEMPORAL FILTER: Reject objects with massive gaps (> 150 days)
    # This filters out "flickering" sources that aren't single explosions
    mjd_sorted = df['mjd'].sort_values().values
    if len(mjd_sorted) > 1 and np.max(np.diff(mjd_sorted)) > 150:
        return None

    res = {"ztf_id": ztf_id, "label": label}

    for fid, band in [(1, 'g'), (2, 'r')]:
        b = df[df['fid'] == fid].sort_values('mjd')

        if len(b) < 3:
            res[f"{band}_peak"] = np.nan
            res[f"{band}_rise"] = np.nan
            res[f"{band}_duration"] = np.nan
            res[f"{band}_stability"] = np.nan
            continue

        mags = b[mag_col].values
        mjds = b['mjd'].values
        flux = mag_to_flux(mags)

        res[f"{band}_peak"] = np.min(mags)
        res[f"{band}_duration"] = mjds[-1] - mjds[0]

        # Rise Rate
        peak_idx = np.argmin(mags)
        if peak_idx > 0:
            dt = mjds[peak_idx] - mjds[0]
            df_flux = flux[peak_idx] - flux[0]
            res[f"{band}_rise"] = df_flux / max(dt, 0.1)
        else:
            res[f"{band}_rise"] = 0

        # Stability: Mean absolute difference
        res[f"{band}_stability"] = np.mean(np.abs(np.diff(mags)))

    # Burst Ratio (Uses the duration we just calculated)
    total_time_span = mjd_sorted[-1] - mjd_sorted[0]
    duration_g = res.get('g_duration', 0)
    duration_r = res.get('r_duration', 0)
    res['burst_ratio'] = max(duration_g, duration_r) / max(total_time_span, 1.0)

    # Peak Color
    if not np.isnan(res.get("g_peak", np.nan)) and not np.isnan(res.get("r_peak", np.nan)):
        res["peak_color"] = res["g_peak"] - res["r_peak"]
    else:
        res["peak_color"] = np.nan

    # Physical Absolute Magnitude
    try:
        z = float(redshift)
    except: z = 0.0
    
    if z > 0.001 and not np.isnan(res.get("g_peak", np.nan)):
        dist_pc = (z * 3e5) / 70 * 1e6 
        res["abs_mag_g"] = res["g_peak"] - 5 * np.log10(dist_pc) + 5
    else:
        res["abs_mag_g"] = np.nan

    return res

def run_features():
    with open("ztf_lcs_all.pkl", "rb") as f:
        all_lcs = pickle.load(f)
    
    bts = pd.read_csv("bts_all_labeled.csv").set_index("ZTFID")

    final_data = []
    for ztf_id, df in all_lcs.items():
        label = bts.loc[ztf_id, "type"] if ztf_id in bts.index else "Unknown"
        z = bts.loc[ztf_id, "redshift"] if ztf_id in bts.index else 0
        
        if isinstance(label, pd.Series): label = label.iloc[0]
        if isinstance(z, pd.Series): z = z.iloc[0]

        feat = extract_features(ztf_id, df, label, z)
        if feat: final_data.append(feat)

    # Use median imputation in the training script, but drop rows that are all NaN here
    df_final = pd.DataFrame(final_data)
    df_final.to_csv("ztf_features_clean.csv", index=False)
    print(f"Processed {len(df_final)} clean objects.")

if __name__ == "__main__":
    run_features()