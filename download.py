from alerce.core import Alerce
import pandas as pd
import pickle
import time

client = Alerce()

def run_download(limit=100):
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
            # Query detections
            df_det = client.query_detections(ztf_id, survey="ztf", format="pandas")

            if df_det is None or df_det.empty:
                continue

            # Check for quality if RB score exists
            if "rb" in df_det.columns:
                df_det = df_det[df_det["rb"] >= 0.5]

            if len(df_det) >= 5:
                all_lcs[ztf_id] = df_det

        except Exception as e:
            print(f"Failed {ztf_id}: {e}")

        if i % 5 == 0: # Print progress every 5 objects
            print(f"Progress: {i}/{len(ztf_ids)} | Collected: {len(all_lcs)}")
        time.sleep(0.1) 

    with open("ztf_lcs_all.pkl", "wb") as f:
        pickle.dump(all_lcs, f)

    print(f"Successfully downloaded: {len(all_lcs)}")

if __name__ == "__main__":
    run_download(limit=100)
            