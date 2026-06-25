import json
import pandas as pd
import numpy as np
from scipy import stats

# Load results
with open("top_anomalies.json", "r") as f:
    iforest_data = json.load(f)

with open("ae_anomalies.json", "r") as f:
    ae_data = json.load(f)

with open("ocsvm_anomalies.json", "r") as f:
    ocsvm_data = json.load(f)

# Convert to DataFrames with normalized ranks
# For IF and OCSVM: lower score = more anomalous
# For AE: higher score = more anomalous

df_if = pd.DataFrame(iforest_data)
df_if["if_rank"] = stats.rankdata(df_if["raw_score"], method="min")  # Lower score = lower rank number = more anomalous
df_if["if_norm"] = (df_if["if_rank"] - 1) / (len(df_if) - 1) if len(df_if) > 1 else 0  # 0 = most anomalous

df_ae = pd.DataFrame(ae_data)
df_ae["ae_rank"] = stats.rankdata(-df_ae["ae_score"], method="min")  # Negative for descending
df_ae["ae_norm"] = (df_ae["ae_rank"] - 1) / (len(df_ae) - 1) if len(df_ae) > 1 else 0

df_ocsvm = pd.DataFrame(ocsvm_data)
df_ocsvm["svm_rank"] = stats.rankdata(df_ocsvm["ocsvm_score"], method="min")
df_ocsvm["svm_norm"] = (df_ocsvm["svm_rank"] - 1) / (len(df_ocsvm) - 1) if len(df_ocsvm) > 1 else 0

# Merge all
merged = df_if[["ztf_id", "label", "raw_score", "primary_feature", "if_norm"]].merge(
    df_ae[["ztf_id", "ae_score", "ae_norm"]], on="ztf_id", how="outer"
).merge(
    df_ocsvm[["ztf_id", "ocsvm_score", "svm_norm"]], on="ztf_id", how="outer"
)

# Fill missing ranks with worst possible (1.0 = not anomalous)
merged["if_norm"] = merged["if_norm"].fillna(1.0)
merged["ae_norm"] = merged["ae_norm"].fillna(1.0)
merged["svm_norm"] = merged["svm_norm"].fillna(1.0)

# Consensus score: average of normalized ranks (lower = more anomalous)
merged["consensus_score"] = (merged["if_norm"] + merged["ae_norm"] + merged["svm_norm"]) / 3

# Vote count: how many models flagged this object
merged["vote_count"] = (
    merged["if_norm"].lt(1.0).astype(int) + 
    merged["ae_norm"].lt(1.0).astype(int) + 
    merged["svm_norm"].lt(1.0).astype(int)
)

# Sort by consensus score then by vote count
merged = merged.sort_values(["vote_count", "consensus_score"], ascending=[False, True])

print("\n--- Final Consensus Anomalies ---")
print("Rank | ID | Type | Votes | Consensus | IF Score | AE Score | SVM Score")
print("-" * 80)

for i, (_, row) in enumerate(merged.head(15).iterrows(), 1):
    print(f"{i:2d}   | {row['ztf_id']} | {str(row['label']):12s} | "
          f"{int(row['vote_count'])}     | {row['consensus_score']:.3f}     | "
          f"{row['raw_score']:.3f if pd.notna(row['raw_score']) else 'N/A':>7s} | "
          f"{row['ae_score']:.3f if pd.notna(row['ae_score']) else 'N/A':>7s} | "
          f"{row['ocsvm_score']:.3f if pd.notna(row['ocsvm_score']) else 'N/A':>8s}")

# Save consensus
merged.to_csv("consensus_anomalies.csv", index=False)
print(f"\nSaved {len(merged)} results to consensus_anomalies.csv")

# High-confidence anomalies: 2+ votes and consensus score < 0.3
high_conf = merged[(merged["vote_count"] >= 2) & (merged["consensus_score"] < 0.3)]
print(f"\nHigh-confidence anomalies (2+ votes, score < 0.3): {len(high_conf)}")
if len(high_conf) > 0:
    print(high_conf[["ztf_id", "label", "vote_count", "consensus_score"]].to_string(index=False))