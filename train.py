import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

def detect():
    df = pd.read_csv("ztf_features_clean.csv")
    feature_cols = [c for c in df.columns if c not in ["ztf_id", "label"]]

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.04, random_state=42)
    df["anomaly_score"] = model.fit_predict(X_scaled)
    df["raw_score"] = model.score_samples(X_scaled)

    # Sort: Most snomalous (lowest raw_score) first
    anomalies = df.sort_values("raw_score").head(10)

    print("\n--- Top Anomalies ---")

    # Interpretation: Find the feature that pushed it into the anomaly zone
    medians = df[feature_cols].median()
    stds = df[feature_cols].std()

    for idx, row in anomalies.iterrows():
        # Find which feature is most sigmas away from the median
        z_scores = (row[feature_cols] - medians) / stds
        top_feature = z_scores.abs().idxmax()

        print(f"ID: {row['ztf_id']} | Type: {row['label']}")
        print(f"   > Primary Reason: {top_feature} is {z_scores[top_feature]:.1f} std des from avg")
        print(f"   > Anomaly Score: {row['raw_score']:.3f}\n") 

if __name__ == "__main__":
    detect()

