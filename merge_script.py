import json
import pandas as pd

# Load results
with open("top_anomalies.json", "r") as f: # Isolation Forest
    iforest = {d['ztf_id']: d['raw_score'] for d in json.load(f)}

with open("ae_anomalies.json", "r") as f: # Autoencoder
    ae = {d['ztf_id']: d['ae_score'] for d in json.load(f)}

with open("ocsvm_anomalies.json", "r") as f: # One-Class SVM
    ocsvm = {d['ztf_id']: d['ocsvm_score'] for d in json.load(f)}

# Find IDs that appear in multiple lists
all_ids = set(list(iforest.keys()) + list(ae.keys()) + list(ocsvm.keys()))

consensus_results = []
for zid in all_ids:
    votes = 0
    if zid in iforest: votes += 1
    if zid in ae: votes += 1
    if zid in ocsvm: votes += 1
    
    consensus_results.append({
        "ztf_id": zid,
        "vote_count": votes,
        "models": [m for m, d in zip(["IF", "AE", "SVM"], [iforest, ae, ocsvm]) if zid in d]
    })

# Sort by most votes first
df_consensus = pd.DataFrame(consensus_results).sort_values("vote_count", ascending=False)

print("\n--- Final Consensus Anomalies ---")
print(df_consensus.head(10))