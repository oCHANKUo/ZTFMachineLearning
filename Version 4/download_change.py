from alerce.core import Alerce
import pandas as pd
import pickle
import time

client = Alerce()


def run_download(limit=1000):

    print("Fetching BTS Catalog...")
    bts = pd.read_csv(
        "https://sites.astro.caltech.edu/ztf/bts/explorer.php?format=csv"
    )

    bts_labeled = bts[bts["type"] != "-"].copy()
    bts_labeled.to_csv("bts_all_labeled.csv", index=False)

    print(f"Total labeled objects: {len(bts_labeled)}")
    print("\nTop Classes:")
    print(bts_labeled["type"].value_counts().head(10))

    ztf_ids = bts_labeled["ZTFID"].dropna().unique()[:limit]

    all_lcs = {}

    rejection_stats = {
        "empty": 0,
        "rb": 0,
        "isdiffpos": 0,
        "magerr": 0,
        "magnitude": 0,
        "detections": 0,
        "position": 0
    }

    print(f"\nDownloading detections for {len(ztf_ids)} objects...\n")

    for i, ztf_id in enumerate(ztf_ids):

        try:

            df_det = client.query_detections(
                ztf_id,
                survey="ztf",
                format="pandas"
            )

            if df_det is None or df_det.empty:
                rejection_stats["empty"] += 1
                continue

            # Print diagnostics only once
            if i == 0:
                print("\nColumns returned:")
                print(df_det.columns.tolist())

                if "isdiffpos" in df_det.columns:
                    print("\nisdiffpos values:")
                    print(df_det["isdiffpos"].unique())

            original_size = len(df_det)

            # -------------------------------------------------
            # REAL-BOGUS FILTER
            # -------------------------------------------------

            if "rb" in df_det.columns:

                before = len(df_det)
                df_det = df_det[df_det["rb"] >= 0.55]

                if len(df_det) == 0:
                    rejection_stats["rb"] += 1
                    continue

            # -------------------------------------------------
            # POSITIVE SUBTRACTION FILTER
            # -------------------------------------------------

            if "isdiffpos" in df_det.columns:

                before = len(df_det)

                positive_values = ["t", "1", True]

                df_det = df_det[
                    df_det["isdiffpos"].isin(positive_values)
                ]

                if len(df_det) == 0:
                    rejection_stats["isdiffpos"] += 1
                    continue

            # -------------------------------------------------
            # MAGNITUDE ERROR FILTER
            # -------------------------------------------------

            if "magerr" in df_det.columns:

                df_det = df_det[
                    (df_det["magerr"] > 0)
                    & (df_det["magerr"] < 0.5)
                ]

                if len(df_det) == 0:
                    rejection_stats["magerr"] += 1
                    continue

            # -------------------------------------------------
            # MAGNITUDE RANGE FILTER
            # -------------------------------------------------

            mag_col = None

            if "magpsf" in df_det.columns:
                mag_col = "magpsf"
            elif "magpsf_corr" in df_det.columns:
                mag_col = "magpsf_corr"

            if mag_col is not None:

                df_det = df_det[
                    (df_det[mag_col] > 12)
                    & (df_det[mag_col] < 25)
                ]

                if len(df_det) == 0:
                    rejection_stats["magnitude"] += 1
                    continue

            # -------------------------------------------------
            # DETECTION COUNT FILTER
            # -------------------------------------------------

            if "fid" in df_det.columns:

                g_count = len(df_det[df_det["fid"] == 1])
                r_count = len(df_det[df_det["fid"] == 2])

            else:

                g_count = 0
                r_count = 0

            if len(df_det) < 8 or g_count < 3 or r_count < 3:
                rejection_stats["detections"] += 1
                continue

            # -------------------------------------------------
            # POSITION CONSISTENCY FILTER
            # -------------------------------------------------

            if "ra" in df_det.columns and "dec" in df_det.columns:

                ra_std = df_det["ra"].std()
                dec_std = df_det["dec"].std()

                if ra_std > 0.01 or dec_std > 0.01:
                    rejection_stats["position"] += 1
                    continue

            # -------------------------------------------------
            # SAVE OBJECT
            # -------------------------------------------------

            all_lcs[ztf_id] = df_det

            if len(all_lcs) % 10 == 0:
                print(f"Collected {len(all_lcs)} good light curves")

        except Exception as e:

            print(f"Failed {ztf_id}: {e}")

        if i % 10 == 0:

            print(
                f"Progress: {i}/{len(ztf_ids)} "
                f"| Collected: {len(all_lcs)}"
            )

        time.sleep(0.05)

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    with open("ztf_lcs_all.pkl", "wb") as f:
        pickle.dump(all_lcs, f)

    print("\n========================")
    print("DOWNLOAD COMPLETE")
    print("========================")

    print(f"Accepted objects: {len(all_lcs)}")

    print("\nRejection Summary:")
    for key, value in rejection_stats.items():
        print(f"{key:12s}: {value}")

    print("\nSaved to ztf_lcs_all.pkl")


if __name__ == "__main__":
    run_download(limit=1000)