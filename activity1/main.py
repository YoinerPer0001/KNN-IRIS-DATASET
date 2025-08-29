from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fetch dataset 
iris = fetch_ucirepo(id=53) 

# Separar características (X) y etiquetas (y)
X = iris.data.features
y = iris.data.targets.values.ravel()  # Convertir a vector plano si es necesario

# Número de vecinos
k = 3

# Nuevos puntos a clasificar
nuevos_puntos = [
    [5.0, 3.5, 1.5, 0.5],
    [5.0, 3.5, 2.0, 0.2],
    [5.0, 2.1, 1.5, 0.1],
    [5.0, 3.5, 1.5, 0.2]
]

# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# Clasificar cada nuevo punto
for nuevo_punto in nuevos_puntos:
    prediccion = knn.predict([nuevo_punto])[0]
    vecinos = knn.kneighbors([nuevo_punto], return_distance=False)[0]

    # Imprimir información
    print(f"Clase predicha para el nuevo punto {nuevo_punto}: Clase {prediccion}")
    print(f"IDs (índices) de los {k} vecinos más cercanos: {vecinos}")
