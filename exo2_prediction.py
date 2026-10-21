import json, warnings
import joblib, sklearn
import pandas as pd

modele = joblib.load("modeles/diabete.joblib")
with open("modeles/diabete_infos.json", encoding="utf-8") as f:
    infos = json.load(f)

if sklearn.__version__ != infos["scikit_learn"]:
    warnings.warn(f"Version différente : modèle sauvegardé avec "
                  f"{infos['scikit_learn']}, poste en {sklearn.__version__}.")
else:
    print("Versions cohérentes : scikit-learn", sklearn.__version__)

variables = infos["variables"]
rmse = infos["metriques_test"]["rmse"]

patient = {"age": 0.038, "sex": 0.051, "bmi": 0.062, "bp": 0.022, "s1": -0.044,
           "s2": -0.035, "s3": -0.043, "s4": -0.003, "s5": 0.020, "s6": -0.018}
valeur = modele.predict(pd.DataFrame([patient])[variables])[0]
print(f"Patient seul : progression estimée à {valeur:.0f}, à ± {rmse:.0f} près")

lot = pd.read_csv("patients.csv")[variables]
for i, v in enumerate(modele.predict(lot), start=1):
    print(f"Patient {i} : progression estimée à {v:.0f}, à ± {rmse:.0f} près")