import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('datasetSportman.csv')

X = df[["Velocidad", "Agilidad"]]
y = df["profesional"]

k = 3

# Crear y entrenar el modelo KNN con k=3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

newSportman = [[6.75,3.00]]

# Clasificar con KNN
prediccion = knn.predict(newSportman)[0]
vecinos = knn.kneighbors(newSportman, return_distance=False)[0]

# Imprimir información
print(f"Clase predicha para el nuevo punto {newSportman[0]}: Clase {prediccion}")
print(f"IDs (índices) de los 3 vecinos más cercanos: {vecinos}")

# Visualizar
plt.figure(figsize=(8, 6))

# Graficar los datos por clase
for target in df['profesional'].unique():
    cluster = df[df['profesional'] == target]
    print(f"cluster = {cluster}")
    plt.scatter(cluster["Velocidad"], cluster["Agilidad"], label=f'Clase {target}')

# Graficar el nuevo punto
plt.scatter(newSportman[0][0], newSportman[0][1], color='black', marker='X', s=100, label='new sportman')

# Graficar los vecinos más cercanos
for idx in vecinos:
    vecino = X.iloc[idx]
    plt.scatter(vecino[0], vecino[1], edgecolor='red', facecolor='none', s=200, linewidths=2)

# Atributos para el gráfico
plt.title(f"Clasificación con KNN k={k}")
plt.xlabel("Velocidad")
plt.ylabel("Agilidad")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


