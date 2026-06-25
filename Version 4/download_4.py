from alerce.core import Alerce
import pandas as pd
import pickle
import time
import numpy as np

client = Alerce()

def run_download(limit=1000):
    print("Fetching BTS Catalog...")
    bts = pd.read_csv("https://sites.astro.caltech.edu/ztf/bts/explorer.php?format=csv")

    bts_labeled = bts[bts["type"] != "-"].copy()
    bts_labeled.to_csv("bts_all_labeled.csv", index=False)

    print(f"Total labeled objects: {len(bts_labeled)}")
    print(bts_labeled["type"].value_counts().head(10))

    ztf_ids = bts_labeled["ZTFID"].dropna().unique()[:limit]
    all_lcs = {}

    print(f"\nDownloading Detections for {len(ztf_ids)} objects...")

    for i, ztf_id in enumerate(ztf_ids):
        try:
            df_det = client.query_detections(ztf_id, survey="ztf", format="pandas")

            if df_det is None or df_det.empty:
                continue

            # QUALITY CUTS
            # 1. Real-bogus score
            if "rb" in df_det.columns:
                df_det = df_det[df_det["rb"] >= 0.55]

            # 2. Positive difference (negative = subtraction artifact)
            if "isdiffpos" in df_det.columns:
                df_det = df_det[df_det["isdiffpos"] == "t"]

            # 3. Reasonable magnitude errors
            if "magerr" in df_det.columns:
                df_det = df_det[df_det["magerr"] < 0.5]
                df_det = df_det[df_det["magerr"] > 0.0]

            # 4. No crazy magnitudes (artifacts often at ~99.99)
            mag_col = "magpsf" if "magpsf" in df_det.columns else "magpsf_corr"
            df_det = df_det[df_det[mag_col] < 25]
            df_det = df_det[df_det[mag_col] > 12]

            # 5. Minimum detections per band
            g_count = len(df_det[df_det["fid"] == 1])
            r_count = len(df_det[df_det["fid"] == 2])

            if len(df_det) >= 8 and g_count >= 3 and r_count >= 3:
                # 6. Spatial consistency (check if coordinates vary wildly)
                if "ra" in df_det.columns and "dec" in df_det.columns:
                    ra_std = df_det["ra"].std()
                    dec_std = df_det["dec"].std()
                    if ra_std > 0.01 or dec_std > 0.01:  # ~36 arcsec
                        continue

                all_lcs[ztf_id] = df_det

        except Exception as e:
            print(f"Failed {ztf_id}: {e}")

        if i % 10 == 0:
            print(f"Progress: {i}/{len(ztf_ids)} | Collected: {len(all_lcs)}")
        time.sleep(0.05)

    with open("ztf_lcs_all.pkl", "wb") as f:
        pickle.dump(all_lcs, f)

    print(f"Successfully downloaded: {len(all_lcs)}")

if __name__ == "__main__":
    run_download(limit=1000)