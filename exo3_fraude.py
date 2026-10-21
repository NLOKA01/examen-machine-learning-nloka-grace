import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, roc_curve, precision_recall_curve)

# Génération
rng = np.random.default_rng(42)
n = 20_000
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

print(f"Taux de fraude : {tx['fraude'].mean():.2%}")

cols = ["log_montant", "nuit", "distance_km", "nb_transactions_24h", "etranger"]
X_tr, X_te, y_tr, y_te = train_test_split(tx[cols], tx["fraude"], test_size=0.25,
                                          stratify=tx["fraude"], random_state=42)

# Modèle naïf
dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print(f"Modèle naïf exactitude : {dummy.score(X_te, y_te):.4f}")

# Régression logistique
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
p_cv = cross_val_predict(logreg, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]

# Seuil optimal sur entraînement
seuils = np.linspace(0.005, 0.95, 400)
couts = [200 * np.sum((p_cv < s) & (y_tr == 1)) + 10 * np.sum((p_cv >= s) & (y_tr == 0))
         for s in seuils]
s_opt = seuils[np.argmin(couts)]
print(f"Seuil optimal : {s_opt:.3f}")

# Confusion au seuil 0.5
y_pred_05 = (p_cv >= 0.5).astype(int)
print("Matrice seuil 0.5 :\n", confusion_matrix(y_tr, y_pred_05))

# Test
logreg.fit(X_tr, y_tr)
p_te = logreg.predict_proba(X_te)[:, 1]
y_te_pred = (p_te >= s_opt).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, y_te_pred).ravel()
print(f"Test — TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Coût modèle = {200*fn + 10*fp}, sans détection = {200*y_te.sum()}")

# Métriques
print(f"Exactitude={accuracy_score(y_te, y_te_pred):.4f}, "
      f"Précision={precision_score(y_te, y_te_pred):.4f}, "
      f"Rappel={recall_score(y_te, y_te_pred):.4f}, "
      f"F1={f1_score(y_te, y_te_pred):.4f}, "
      f"ROC-AUC={roc_auc_score(y_te, p_te):.4f}, "
      f"PR-AUC={average_precision_score(y_te, p_te):.4f}, "
      f"MCC={matthews_corrcoef(y_te, y_te_pred):.4f}")

# Courbes
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fpr, tpr, _ = roc_curve(y_te, p_te)
ax[0].plot(fpr, tpr); ax[0].plot([0,1],[0,1],"r--")
ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
pr, rc, _ = precision_recall_curve(y_te, p_te)
ax[1].plot(rc, pr); ax[1].set_title("Précision-Rappel")
ax[1].set_xlabel("Rappel"); ax[1].set_ylabel("Précision")
plt.savefig("ex3_roc_pr.png"); plt.show()