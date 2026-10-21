import json, joblib, sys, sklearn
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

os.makedirs("modeles", exist_ok=True)

# 1. Chargement
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
print("X.shape:", X.shape, "| y.shape:", y.shape)
print("Variables:", list(X.columns))

# 2. Découpage AVANT transformation
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pipeline
pipe = make_pipeline(StandardScaler(), LinearRegression())
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. CV
scoring = {"mae": "neg_mean_absolute_error",
           "rmse": "neg_root_mean_squared_error", "r2": "r2"}
res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring)
print(f"CV — MAE={-res['test_mae'].mean():.2f}, "
      f"RMSE={-res['test_rmse'].mean():.2f}, R²={res['test_r2'].mean():.4f}")

# 5. Ridge
grid = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                    {"ridge__alpha": np.logspace(-3, 3, 25)},
                    cv=cv, scoring="neg_root_mean_squared_error")
grid.fit(X_tr, y_tr)
print(f"Ridge alpha={grid.best_params_['ridge__alpha']:.3f}, "
      f"RMSE CV={-grid.best_score_:.2f}")

# 6. Test
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
mae = mean_absolute_error(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
r2 = r2_score(y_te, y_pred)
print(f"Test — MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

# Figure
plt.scatter(y_te, y_pred, alpha=0.6)
lim = [y_te.min(), y_te.max()]; plt.plot(lim, lim, "r--")
plt.xlabel("Valeur réelle"); plt.ylabel("Valeur prédite")
plt.title("Prédit contre réel (test)"); plt.grid(True)
plt.savefig("ex2_predit_reel.png"); plt.show()

# 7. Sauvegarde
joblib.dump(pipe, "modeles/diabete.joblib")
infos = {
    "date": str(date.today()),
    "python": sys.version.split()[0],
    "scikit_learn": sklearn.__version__,
    "joblib": joblib.__version__,
    "variables": list(X.columns),
    "metriques_test": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
}
with open("modeles/diabete_infos.json", "w", encoding="utf-8") as f:
    json.dump(infos, f, indent=2)
print("Sauvegarde OK.")