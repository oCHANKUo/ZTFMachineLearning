import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def detect():
    df = pd.read_csv("ztf_features_clean.csv")
    feature_cols = [c for c in df.columns if c not in ["ztf_id", "label"]]

    # Separate numeric features only
    X = df[feature_cols].select_dtypes(include=[np.number])
    numeric_cols = X.columns.tolist()

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train/validation split for calibration
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

    # Use validation set to calibrate contamination
    # Try multiple contamination values and pick one that gives reasonable validation scores
    best_model = None
    best_contamination = 0.05

    for cont in [0.02, 0.05, 0.08, 0.10]:
        model = IsolationForest(contamination=cont, random_state=42, n_estimators=200)
        model.fit(X_train)
        val_scores = model.score_samples(X_val)
        # We want contamination where validation scores are well-distributed
        n_anom = np.sum(val_scores < np.percentile(val_scores, cont * 100))
        if abs(n_anom / len(val_scores) - cont) < 0.02:
            best_contamination = cont
            best_model = model

    if best_model is None:
        best_model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
        best_model.fit(X_scaled)

    # Predict on full dataset
    df["anomaly_label"] = best_model.predict(X_scaled)
    df["raw_score"] = best_model.score_samples(X_scaled)

    # Get feature importances (approximate using tree paths)
    # For IF, we use the deviation from median as interpretation
    medians = np.median(X_scaled, axis=0)

    # Sort: Most anomalous (lowest raw_score) first
    anomalies = df.sort_values("raw_score").head(15)

    print(f"\n--- Isolation Forest (contamination={best_contamination}) ---")

    results = []
    for idx, row in anomalies.iterrows():
        # Find which features are most unusual
        x_scaled = scaler.transform(imputer.transform([row[numeric_cols].values]))[0]
        deviations = np.abs(x_scaled - medians)
        top_idx = np.argmax(deviations)
        top_feature = numeric_cols[top_idx]
        deviation = deviations[top_idx]

        # Get top 3 unusual features
        top3_idx = np.argsort(deviations)[-3:][::-1]
        top3_features = [(numeric_cols[i], deviations[i]) for i in top3_idx]

        print(f"ID: {row['ztf_id']} | Type: {row['label']}")
        print(f"   > Score: {row['raw_score']:.3f} | Primary: {top_feature} ({deviation:.1f}σ)")
        print(f"   > Top 3: {', '.join([f'{f}({d:.1f}σ)' for f, d in top3_features])}")

        results.append({
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "raw_score": float(row["raw_score"]),
            "primary_feature": top_feature,
            "deviation": float(deviation),
            "top3_features": top3_features
        })

    with open("top_anomalies.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} anomalies to top_anomalies.json")

if __name__ == "__main__":
    detect()