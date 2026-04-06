import pandas as pd
import pickle
import numpy as np

def mag_to_flux(mag):
        # Standard ZTF zeropoint is roughly 25 or 26.3, but 25 works for relative rates
    return 10**(-0.4 * (mag - 25.0))

def extract_features(ztf_id, df, label, redshift):
    # ALeRCE sometimes uses 'magpsf' and sometimes 'magpsf_corr'
    # This checks which one is available
    mag_col = 'magpsf' if 'magpsf' in df.columns else 'magpsf_corr'

    res = {"ztf_id": ztf_id, "label": label}

    for fid, band in [(1, 'g'), (2, 'r')]:
        # Filter detections for specific band
        b = df[df['fid'] == fid].sort_values('mjd')

        # Taking at least 3 points
        if len(b) < 3:
            res[f"{band}_peak"] = np.nan
            res[f"{band}_rise"] = np.nan
            res[f"{band}_duration"] = np.nan
            continue

        mags = b[mag_col].values
        mjds = b['mjd'].values
        flux = mag_to_flux(mags)

        res[f"{band}_peak"] = np.min(mags)
        res[f"{band}_duration"] = mjds[-1] - mjds[0]

        # Calculate Rise Rate (change in flux per day from start to peak)
        peak_idx = np.argmin(mags)
        if peak_idx > 0:
            dt = mjds[peak_idx] - mjds[0]
            df_flux = flux[peak_idx] - flux[0]
            res[f"{band}_rise"] = df_flux / max(dt, 0.1)  # Avoid division by zero
        else:
            res[f"{band}_rise"] = 0

    try:
        z = float(redshift)
    except (ValueError, TypeError):
        z = 0.0

    # Physical Feature: Peak absolute magnitute
    if z > 0.001 and not np.isnan(res.get("g_peak", np.nan)):
        # Formula: M = m - 5*log10(d_L) - 25 (approximating distance with redshift)
        # Using Hubble's Law: d = (z * c) / H0
        dist_pc = (z * 3e5) / 70 * 1e6 # distance in parsecs
        res["abs_mag_g"] = res["g_peak"] - 5 * np.log10(dist_pc) + 5
    else:
        res["abs_mag_g"] = np.nan

    return res

def run_features():
    print("Loading light curves...")
    with open("ztf_lcs_all.pkl", "rb") as f:
              all_lcs = pickle.load(f)
    
    print("Loading BTS Labels...")
    bts = pd.read_csv("bts_all_labeled.csv").set_index("ZTFID")

    final_data = []
    for ztf_id, df in all_lcs.items():
         # Get label and reshift from BTS catalog if it exists
         label = bts.loc[ztf_id, "type"] if ztf_id in bts.index else "Unknown"
         z = bts.loc[ztf_id, "redshift"] if ztf_id in bts.index else 0

         # If BTS has multiple entries for one ID, loc returns a Series. We take the first one.
         if isinstance(label, pd.Series): label = label.iloc[0]
         if isinstance(z, pd.Series): z = z.iloc[0]

         feat = extract_features(ztf_id, df, label, z)
         final_data.append(feat)

    
    # Create DataFrame and drop rows with missing critical features
    df_final = pd.DataFrame(final_data).dropna()

    if df_final.empty:
         print("Error: No valid features extracted. Check the input data. (Check if g_peak or r_peak are all NaN.)")
    else:
         df_final.to_csv("ztf_features_clean.csv", index=False)
         print(f"Feature extraction complete. Processed {len(df_final)} objects")


if __name__ == "__main__":
     run_features()