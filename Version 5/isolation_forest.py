import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def detect():
    # Use STRICT features (clean data only)
    df = pd.read_csv("ztf_features_strict.csv")

    # Define ROBUST feature set (insensitive to sampling)
    # These are features that don't depend on having complete light curves
    robust_features = [
        # Peak properties (single measurement)
        "g_peak", "r_peak",
        "peak_color",
        "abs_mag_g",

        # Simple global properties
        "total_span",
        "n_detections",

        # Data quality (as explicit inputs - model learns these are "normal")
        "dq_total_points",
        "dq_span",
        "dq_sampling_density",
        "dq_log_n",
    ]

    # Also include shape features if object has enough data
    # But we already filtered for that in strict mode
    shape_features = [
        "g_skew", "r_skew",
        "g_kurt", "r_kurt",
        "g_asymmetry", "r_asymmetry",
    ]

    feature_cols = [c for c in robust_features + shape_features if c in df.columns]

    print(f"Using {len(feature_cols)} robust features:")
    for c in feature_cols:
        print(f"  {c}")

    X = df[feature_cols].select_dtypes(include=[np.number])
    numeric_cols = X.columns.tolist()

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train/validation split
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

    # Calibrate contamination
    best_model = None
    best_contamination = 0.05

    for cont in [0.02, 0.05, 0.08, 0.10, 0.15]:
        model = IsolationForest(contamination=cont, random_state=42, n_estimators=200)
        model.fit(X_train)
        val_scores = model.score_samples(X_val)
        n_anom = np.sum(val_scores < np.percentile(val_scores, cont * 100))
        if abs(n_anom / len(val_scores) - cont) < 0.03:
            best_contamination = cont
            best_model = model
            break

    if best_model is None:
        best_model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
        best_model.fit(X_scaled)

    df["anomaly_label"] = best_model.predict(X_scaled)
    df["raw_score"] = best_model.score_samples(X_scaled)

    medians = np.median(X_scaled, axis=0)

    anomalies = df.sort_values("raw_score").head(15)

    print(f"\n--- Isolation Forest (contamination={best_contamination}) ---")
    print(f"Dataset: {len(df)} objects (strict quality cuts applied)")

    results = []
    for idx, row in anomalies.iterrows():
        x_scaled = scaler.transform(imputer.transform([row[numeric_cols].values]))[0]
        deviations = np.abs(x_scaled - medians)
        top_idx = np.argmax(deviations)
        top_feature = numeric_cols[top_idx]
        deviation = deviations[top_idx]

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

    with open("top_anomalies_strict.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} anomalies to top_anomalies_strict.json")

if __name__ == "__main__":
    detect()