from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# K-Means es un algoritmo que agrupa datos en un número predefinido de clústeres (grupos) basándose en su similitud. Cada dato pertenece al clúster cuyo centroide (centro del grupo) está más cerca.

# Cargar el dataset ajustado
df = pd.read_csv("dataset_viajes.csv")

# Codificar las variables categóricas (Inicio y Destino)
label_encoder = LabelEncoder()
df["Inicio"] = label_encoder.fit_transform(df["Inicio"])
df["Destino"] = label_encoder.fit_transform(df["Destino"])

# Seleccionar características para clustering
X = df[["Inicio", "Destino", "Tiempo"]]

# Escalar los datos para que todas las características tengan mismo peso(todas las características tengan una contribución igual en el cálculo de distancias.)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means con 3 clústeres (3 grupos)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualizar los resultados
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 2], c=df["Cluster"], cmap="viridis", s=50)
plt.xlabel("Inicio (escalado)")
plt.ylabel("Tiempo (escalado)")
plt.title("Clustering de Rutas")
plt.colorbar(label="Cluster")
plt.show()

# Mostrar ejemplos de cada clúster
for cluster_id in range(3):
    print(f"\nRutas en el clúster {cluster_id}:")
    print(df[df["Cluster"] == cluster_id].head())
