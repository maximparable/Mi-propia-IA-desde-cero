import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

np.random.seed(42)

data = pd.DataFrame({
    'horas_estudio': np.random.randint(1, 20, 100),
    'promedio': np.random.uniform(2.0, 5.0, 100),
    'materias_perdidas': np.random.randint(0, 5, 100),
    'edad': np.random.randint(18, 40, 100),
})

data['abandono'] = (
    (data['promedio'] < 3.0) &
    (data['materias_perdidas'] > 2)
).astype(int)

X = data.drop('abandono', axis=1)
y = data['abandono']

modelo = RandomForestClassifier()
modelo.fit(X, y)

with open("modelo.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("✅ Modelo creado correctamente")