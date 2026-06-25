import pandas as pd
import numpy as np
import json
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def detect_ocsvm():
    df = pd.read_csv("ztf_features_clean.csv")
    feature_cols = [c for c in df.columns if c not in ["ztf_id", "label"]]

    X = df[feature_cols].select_dtypes(include=[np.number])
    numeric_cols = X.columns.tolist()

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train/val split
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

    # Grid search for best nu and gamma
    best_nu = 0.05
    best_gamma = 'scale'
    best_score = -np.inf

    for nu in [0.03, 0.05, 0.08]:
        for gamma in ['scale', 'auto', 0.01]:
            model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            model.fit(X_train)
            val_scores = model.decision_function(X_val)
            # Higher median score = better separation
            median_score = np.median(val_scores)
            if median_score > best_score:
                best_score = median_score
                best_nu = nu
                best_gamma = gamma

    # Final model
    model = OneClassSVM(kernel='rbf', nu=best_nu, gamma=best_gamma)
    model.fit(X_scaled)

    df["anomaly_label"] = model.predict(X_scaled)
    df["ocsvm_score"] = model.decision_function(X_scaled)

    # Interpretation: distance from decision boundary
    # More negative = more anomalous
    medians = np.median(X_scaled, axis=0)

    anomalies = df.sort_values("ocsvm_score").head(15)

    print(f"\n--- One-Class SVM (nu={best_nu}, gamma={best_gamma}) ---")

    results = []
    for idx, row in anomalies.iterrows():
        x_scaled = scaler.transform(imputer.transform([row[numeric_cols].values]))[0]

        # Distance from center in scaled space
        distances = np.abs(x_scaled - medians)
        top_idx = np.argmax(distances)
        top_feature = numeric_cols[top_idx]

        # Top 3
        top3_idx = np.argsort(distances)[-3:][::-1]
        top3_features = [(numeric_cols[i], distances[i]) for i in top3_idx]

        print(f"ID: {row['ztf_id']} | Type: {row['label']}")
        print(f"   > Distance: {row['ocsvm_score']:.4f} | Extreme: {top_feature}")
        print(f"   > Top 3: {', '.join([f'{f}({d:.2f})' for f, d in top3_features])}")

        results.append({
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "ocsvm_score": float(row["ocsvm_score"]),
            "primary_feature": top_feature,
            "top3_features": top3_features
        })

    with open("ocsvm_anomalies.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} anomalies to ocsvm_anomalies.json")

if __name__ == "__main__":
    detect_ocsvm()