import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reconstitution du jeu de données
rng = np.random.default_rng(42)
surface = rng.uniform(20, 150, size=60)
loyer = 12 * surface + 150 + rng.normal(0, 120, size=60)

# 1. Nuage de points
plt.figure(figsize=(8, 5))
plt.scatter(surface, loyer, alpha=0.7)
plt.xlabel("Surface (m²)"); plt.ylabel("Loyer (€)")
plt.title("Loyer en fonction de la surface"); plt.grid(True)
plt.savefig("ex1_nuage.png"); plt.show()

# Fonctions NumPy
def predire(x, w, b): return w * x + b
def cout_mse(y_vrai, y_predit): return np.mean((y_vrai - y_predit) ** 2)


X = np.column_stack([np.ones_like(surface), surface])
theta = np.linalg.inv(X.T @ X) @ X.T @ loyer
b_exact, w_exact = theta
print(f"Moindres carrés : w={w_exact:.4f}, b={b_exact:.4f}")

model_sk = LinearRegression().fit(surface.reshape(-1, 1), loyer)
print(f"Scikit-learn   : w={model_sk.coef_[0]:.4f}, b={model_sk.intercept_:.4f}")

# Descente de gradient
mu, sd = surface.mean(), surface.std()
ym, ysd = loyer.mean(), loyer.std()
xs, ys = (surface - mu) / sd, (loyer - ym) / ysd

def descente(x, y, lr=0.1, n_iter=200):
    w, b, couts = 0.0, 0.0, []
    n = len(x)
    for _ in range(n_iter):
        y_pred = w * x + b
        err = y_pred - y
        couts.append(np.mean(err**2))
        w -= lr * (2/n) * np.dot(err, x)
        b -= lr * (2/n) * np.sum(err)
    return w, b, np.array(couts)

w_s, b_s, couts = descente(xs, ys, lr=0.1)
w_gd = w_s * ysd / sd
b_gd = ym + ysd * b_s - w_gd * mu
print(f"Descente de gradient : w={w_gd:.4f}, b={b_gd:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(couts); plt.xlabel("Itération"); plt.ylabel("Coût (MSE)")
plt.title("Convergence de la descente de gradient"); plt.grid(True)
plt.savefig("ex1_cout.png"); plt.show()

# 5. Trois taux d'apprentissage
plt.figure(figsize=(8, 4))
for lr, label, c in [(0.01, "0.01 (lent)", 'blue'),
                     (0.3, "0.3 (bon)", 'green'),
                     (1.05, "1.05 (diverge)", 'red')]:
    _, _, h = descente(xs, ys, lr=lr)
    plt.plot(h, label=label, color=c)
plt.yscale('log'); plt.xlabel("Itération"); plt.ylabel("Coût (log)")
plt.legend(); plt.title("Effet du taux d'apprentissage"); plt.grid(True)
plt.savefig("ex1_taux.png"); plt.show()

# Différences finies
def grad_ana(x, y, w, b):
    e = w*x + b - y; return (2/len(x))*np.dot(e, x), (2/len(x))*np.sum(e)
def grad_num(x, y, w, b, eps=1e-6):
    dw = (cout_mse(y, predire(x, w+eps, b)) - cout_mse(y, predire(x, w-eps, b))) / (2*eps)
    db = (cout_mse(y, predire(x, w, b+eps)) - cout_mse(y, predire(x, w, b-eps))) / (2*eps)
    return dw, db

w0, b0 = 0.5, 0.2
ga = grad_ana(xs, ys, w0, b0); gn = grad_num(xs, ys, w0, b0)
print(f"Ecart maximal : {max(abs(ga[0]-gn[0]), abs(ga[1]-gn[1])):.2e}")

# Métriques
y_pred = predire(surface, w_exact, b_exact)
print(f"MAE={mean_absolute_error(loyer, y_pred):.2f}, "
      f"RMSE={np.sqrt(mean_squared_error(loyer, y_pred)):.2f}, "
      f"R²={r2_score(loyer, y_pred):.4f}")


# Prédictions du modèle 
y_pred = predire(surface, w_exact, b_exact)

# Calcul des métriques
mae = mean_absolute_error(loyer, y_pred)
rmse = np.sqrt(mean_squared_error(loyer, y_pred))
r2 = r2_score(loyer, y_pred)

print("\n Les métriques")
print(f"MAE  (Mean Absolute Error)  : {mae:.2f} €")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} €")
print(f"R²   (Coefficient de détermination) : {r2:.4f}")