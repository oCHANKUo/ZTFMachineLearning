import pandas as pd
import pickle
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

def mag_to_flux(mag):
    return 10**(-0.4 * (mag - 25.0))

def sigma_clip(mags, mjds, magerrs, sigma=3.0):
    """Remove 3-sigma outliers from light curve."""
    if len(mags) < 5:
        return mags, mjds, magerrs

    z_scores = np.abs(stats.zscore(mags))
    mask = z_scores < sigma

    # Don't clip the peak (brightest point) - it's likely real
    peak_idx = np.argmin(mags)
    mask[peak_idx] = True

    return mags[mask], mjds[mask], magerrs[mask]

def extract_features(ztf_id, df, label, redshift):
    # 1. QUALITY FILTER
    if 'magerr' in df.columns:
        df = df[df['magerr'] < 0.3]
        df = df[df['magerr'] > 0.0]

    # 2. Remove duplicate MJDs (keep best measurement)
    df = df.sort_values('magerr').drop_duplicates(subset=['mjd', 'fid'], keep='first')

    # 3. Need minimum data
    if len(df) < 10:
        return None

    # 4. Check for crazy magnitudes
    mag_col = 'magpsf' if 'magpsf' in df.columns else 'magpsf_corr'
    if df[mag_col].max() > 24 or df[mag_col].min() < 12:
        return None

    # 5. Temporal sanity - allow some gaps but not everything
    mjd_sorted = df['mjd'].sort_values().values
    total_span = mjd_sorted[-1] - mjd_sorted[0]

    if total_span < 1:  # Less than 1 day
        return None

    # 6. Check for massive single gap (indicates two separate events)
    if len(mjd_sorted) > 1:
        gaps = np.diff(mjd_sorted)
        if np.max(gaps) > 200:  # More lenient than 150
            return None

    res = {"ztf_id": ztf_id, "label": label}

    # Global features
    res["total_span"] = total_span
    res["n_detections"] = len(df)

    # Per-band features
    all_peaks = []
    all_durations = []

    for fid, band in [(1, 'g'), (2, 'r')]:
        b = df[df['fid'] == fid].sort_values('mjd')

        if len(b) < 3:
            # Fill with NaN for missing band
            for suffix in ['peak', 'rise', 'fall', 'duration', 'stability', 
                          'skew', 'kurt', 'nobs', 'slope_pre', 'slope_post',
                          'time_to_peak', 'asymmetry', 'wmean']:
                res[f"{band}_{suffix}"] = np.nan
            continue

        mags = b[mag_col].values
        mjds = b['mjd'].values
        magerrs = b['magerr'].values if 'magerr' in b.columns else np.ones_like(mags) * 0.1

        # Sigma clip outliers
        mags, mjds, magerrs = sigma_clip(mags, mjds, magerrs)

        if len(mags) < 3:
            for suffix in ['peak', 'rise', 'fall', 'duration', 'stability', 
                          'skew', 'kurt', 'nobs', 'slope_pre', 'slope_post',
                          'time_to_peak', 'asymmetry', 'wmean']:
                res[f"{band}_{suffix}"] = np.nan
            continue

        flux = mag_to_flux(mags)

        # Basic features
        peak_idx = np.argmin(mags)
        peak_mag = mags[peak_idx]
        peak_mjd = mjds[peak_idx]

        res[f"{band}_peak"] = peak_mag
        res[f"{band}_duration"] = mjds[-1] - mjds[0]
        res[f"{band}_nobs"] = len(mags)
        res[f"{band}_wmean"] = np.average(mags, weights=1.0/magerrs)

        all_peaks.append(peak_mag)
        all_durations.append(res[f"{band}_duration"])

        # Rise (before peak)
        pre_peak = mags[:peak_idx+1]
        pre_mjds = mjds[:peak_idx+1]
        if len(pre_peak) >= 2:
            dt = pre_mjds[-1] - pre_mjds[0]
            dflux = flux[peak_idx] - flux[0]
            res[f"{band}_rise"] = dflux / max(dt, 0.1)
            res[f"{band}_time_to_peak"] = dt

            # Linear slope before peak
            if len(pre_peak) >= 3:
                slope, _, _, _, _ = stats.linregress(pre_mjds, pre_peak)
                res[f"{band}_slope_pre"] = slope
            else:
                res[f"{band}_slope_pre"] = np.nan
        else:
            res[f"{band}_rise"] = 0
            res[f"{band}_time_to_peak"] = 0
            res[f"{band}_slope_pre"] = np.nan

        # Fall (after peak)
        post_peak = mags[peak_idx:]
        post_mjds = mjds[peak_idx:]
        if len(post_peak) >= 2:
            dt = post_mjds[-1] - post_mjds[0]
            dflux = flux[-1] - flux[peak_idx]
            res[f"{band}_fall"] = dflux / max(dt, 0.1)

            if len(post_peak) >= 3:
                slope, _, _, _, _ = stats.linregress(post_mjds, post_peak)
                res[f"{band}_slope_post"] = slope
            else:
                res[f"{band}_slope_post"] = np.nan
        else:
            res[f"{band}_fall"] = 0
            res[f"{band}_slope_post"] = np.nan

        # Shape statistics
        res[f"{band}_stability"] = np.mean(np.abs(np.diff(mags)))
        res[f"{band}_skew"] = stats.skew(mags)
        res[f"{band}_kurt"] = stats.kurtosis(mags)

        # Asymmetry: ratio of rise time to fall time
        rise_time = peak_mjd - mjds[0] if peak_idx > 0 else 0.1
        fall_time = mjds[-1] - peak_mjd if peak_idx < len(mjds)-1 else 0.1
        res[f"{band}_asymmetry"] = rise_time / max(fall_time, 0.1)

    # Cross-band features
    if len(all_peaks) == 2:
        res["peak_color"] = all_peaks[0] - all_peaks[1]  # g - r
    else:
        res["peak_color"] = np.nan

    # Duration ratio
    if len(all_durations) == 2 and all_durations[1] > 0:
        res["duration_ratio"] = all_durations[0] / all_durations[1]
    else:
        res["duration_ratio"] = np.nan

    # Peak color evolution (if we have enough points in both bands)
    g_data = df[df['fid'] == 1].sort_values('mjd')
    r_data = df[df['fid'] == 2].sort_values('mjd')

    if len(g_data) >= 3 and len(r_data) >= 3:
        # Interpolate to common grid and measure color change
        common_mjds = np.linspace(
            max(g_data['mjd'].min(), r_data['mjd'].min()),
            min(g_data['mjd'].max(), r_data['mjd'].max()),
            20
        )

        try:
            g_interp = interp1d(g_data['mjd'], g_data[mag_col], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
            r_interp = interp1d(r_data['mjd'], r_data[mag_col], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')

            g_common = g_interp(common_mjds)
            r_common = r_interp(common_mjds)
            colors = g_common - r_common

            res["color_slope"] = np.polyfit(common_mjds, colors, 1)[0] if len(common_mjds) > 2 else np.nan
            res["color_range"] = np.max(colors) - np.min(colors)
        except:
            res["color_slope"] = np.nan
            res["color_range"] = np.nan
    else:
        res["color_slope"] = np.nan
        res["color_range"] = np.nan

    # Absolute magnitude
    try:
        z = float(redshift)
    except:
        z = 0.0

    if z > 0.001 and not np.isnan(res.get("g_peak", np.nan)):
        # Use proper cosmology approximation
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
    failed = 0

    for ztf_id, df in all_lcs.items():
        label = bts.loc[ztf_id, "type"] if ztf_id in bts.index else "Unknown"
        z = bts.loc[ztf_id, "redshift"] if ztf_id in bts.index else 0

        if isinstance(label, pd.Series): 
            label = label.iloc[0]
        if isinstance(z, pd.Series): 
            z = z.iloc[0]

        feat = extract_features(ztf_id, df, label, z)
        if feat:
            final_data.append(feat)
        else:
            failed += 1

    df_final = pd.DataFrame(final_data)

    # Drop columns that are entirely NaN
    df_final = df_final.dropna(axis=1, how='all')

    df_final.to_csv("ztf_features_clean.csv", index=False)
    print(f"Processed {len(df_final)} clean objects (rejected {failed}).")
    print(f"Features: {list(df_final.columns)}")

if __name__ == "__main__":
    run_features()