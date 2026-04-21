# 3_train_autoencoder.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
import json

# 1. Prepare Data
df = pd.read_csv("ztf_features_clean.csv")
feature_cols = [c for c in df.columns if c not in ["ztf_id", "label"]]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
X_tensor = torch.FloatTensor(X_scaled)

# 2. Define the Architecture
class AnomalyAE(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.BatchNorm1d(12), # Helps stabilize training
            nn.ReLU(),
            nn.Linear(12, 6) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. Training Loop
input_dim = len(feature_cols)
model = AnomalyAE(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training Autoencoder...")
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 4. Scoring Anomalies (Reconstruction Error)
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    # Using Absolute Error instead of Squared Error
    reconstruction_errors = torch.mean(torch.abs(X_tensor - predictions), dim=1).numpy()
    
df["anomaly_score"] = reconstruction_errors

# 5. Interpret and Save
# In AE, HIGHER score = MORE anomalous
anomalies = df.sort_values("anomaly_score", ascending=False).head(10)

print("\n--- Top Autoencoder Anomalies ---")
results = []
for idx, row in anomalies.iterrows():
    print(f"ID: {row['ztf_id']} | Type: {row['label']}")
    print(f"   > Reconstruction Error: {row['anomaly_score']:.6f}\n")
    
    results.append({
        "ztf_id": row["ztf_id"],
        "label": row["label"],
        "ae_score": float(row["anomaly_score"])
    })

with open("ae_anomalies.json", "w") as f:
    json.dump(results, f, indent=2)