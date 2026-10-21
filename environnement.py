import sys
import importlib.metadata as md

libs = ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy", "joblib"]

print(f"Version de Python : {sys.version.split()[0]}")
print("-" * 30)

for lib in libs:
    try:
        version = md.version(lib)
        print(f"{lib:15s} : {version}")
    except md.PackageNotFoundError:
        print(f"{lib:15s} : NON INSTALLÉE")