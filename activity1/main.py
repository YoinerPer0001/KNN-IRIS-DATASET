from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
  

# fetch dataset 
iris = fetch_ucirepo(id=53) 

#numero de vecinos
k = 3

#nuevo punto a clasificar
nuevo_punto = [5.0, 3.5, 1.5, 0.2]

# Graficar dataset en 2D
plt.figure(figsize=(8, 6))

X = iris.data.features 
y = iris.data.targets

df = pd.DataFrame({
    "Característica 1": X.iloc[:, 0],
    "Característica 2": X.iloc[:, 1],
    "Target": y.iloc[:, 0]
})



# Crear y entrenar el modelo KNN con
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)


prediccion = knn.predict([nuevo_punto])[0]
vecinos = knn.kneighbors([nuevo_punto], return_distance=False)[0]

# Imprimir información
print(f"Clase predicha para el nuevo punto {nuevo_punto}: Clase {prediccion}")
print(f"IDs (índices) de los 3 vecinos más cercanos: {vecinos}")
