# 3_train_ocsvm.py
import pandas as pd
import numpy as np
import json
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

def detect_ocsvm():
    # 1. Load and Scale Data
    df = pd.read_csv("ztf_features_clean.csv")
    feature_cols = [c for c in df.columns if c not in ["ztf_id", "label"]]

    X = df[feature_cols]

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 2. Train One-Class SVM
    # nu: An upper bound on the fraction of training errors (approx. contamination)
    # kernel='rbf': Uses a radial basis function to handle non-linear boundaries
    # gamma: Defines how far the influence of a single training example reaches
    model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    
    # Fit and Predict (-1 for anomaly, 1 for normal)
    df["anomaly_label"] = model.fit_predict(X_scaled)
    
    # Decision function: Signed distance to the separating hyperplane
    # Lower (more negative) values are more anomalous
    df["ocsvm_score"] = model.decision_function(X_scaled)

    # 3. Sort and Interpret
    anomalies = df.sort_values("ocsvm_score").head(10)

    print("\n--- Top One-Class SVM Anomalies ---")
    medians = df[feature_cols].median()
    stds = df[feature_cols].std()

    results = []
    for idx, row in anomalies.iterrows():
        # Identify the most extreme feature for reporting
        z_scores = (row[feature_cols] - medians) / stds
        top_feature = z_scores.abs().idxmax()

        print(f"ID: {row['ztf_id']} | Type: {row['label']}")
        print(f"   > Distance from Boundary: {row['ocsvm_score']:.4f}")
        print(f"   > Extreme Feature: {top_feature} ({z_scores[top_feature]:.1f} std devs)\n")

        results.append({
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "ocsvm_score": float(row["ocsvm_score"]),
            "primary_feature": top_feature
        })

    with open("ocsvm_anomalies.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    detect_ocsvm()