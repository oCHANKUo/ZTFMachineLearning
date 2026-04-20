# 4_visualize.py
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# List the IDs you want to inspect from your train.py output
top_anomalies = ["ZTF19aacgslb", "ZTF18abcfcoo", "ZTF17aaazdba"]

with open("ztf_lcs_all.pkl", "rb") as f:
    lcs = pickle.load(f)

for ztf_id in top_anomalies:
    if ztf_id not in lcs: continue
    
    df = lcs[ztf_id]
    plt.figure(figsize=(10, 5))
    
    # Plot Green and Red bands
    for fid, color, label in [(1, 'green', 'g-band'), (2, 'red', 'r-band')]:
        band = df[df['fid'] == fid]
        # Use the corrected magnitude we identified earlier
        mag_col = 'magpsf_corr' if 'magpsf_corr' in df.columns else 'magpsf'
        
        plt.scatter(band['mjd'], band[mag_col], c=color, label=label)
    
    plt.gca().invert_yaxis() # Magnitudes are upside down! (Smaller is brighter)
    plt.title(f"Anomaly: {ztf_id}")
    plt.xlabel("Days (MJD)")
    plt.ylabel("Brightness (Mag)")
    plt.legend()
    plt.savefig(f"anomaly_{ztf_id}.png")
    print(f"Saved plot for {ztf_id}")