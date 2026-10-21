import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42); n = 20_000
tx = pd.DataFrame({
    "montant": rng.lognormal(3.5, 1.0, n).round(2),
    "heure": rng.integers(0, 24, n),
    "distance_km": rng.exponential(10, n).round(1),
    "nb_transactions_24h": rng.poisson(2, n),
    "etranger": (rng.random(n) < 0.05).astype(int),
})
nuit = ((tx["heure"] < 6) | (tx["heure"] >= 23)).astype(int)
logit = (-11.5 + np.log(tx["montant"]) + 2.0*nuit + 0.08*tx["distance_km"]
         + 0.5*tx["nb_transactions_24h"] + 3.0*tx["etranger"])
tx["fraude"] = (rng.random(n) < 1/(1+np.exp(-logit))).astype(int)
tx["log_montant"] = np.log(tx["montant"]); tx["nuit"] = nuit

cols = ["log_montant", "nuit", "distance_km", "nb_transactions_24h", "etranger"]
model = LogisticRegression(max_iter=1000).fit(tx[cols], tx["fraude"])
for c, coef in zip(cols, model.coef_[0]):
    print(f"{c:20s} coef={coef:.3f}, OR={np.exp(coef):.2f}")