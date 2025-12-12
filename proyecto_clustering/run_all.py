from src.preprocessing import load_dataset
from src.kmeans_model import run_kmeans, find_optimal_k
from src.hierarchical_model import run_hierarchical
from src.dbscan_model import run_dbscan
from src.evaluation import evaluate_clustering
from src.visualization import (
plot_clusters,
plot_elbow,
plot_silhouette,
)


# 1. Cargar datos
print("Cargando dataset...")
df, X_scaled = load_dataset()
print(df.head())


# 2. Buscar mejor K para K-Means
print("Calculando K óptimo...")
inertias, silhouettes = find_optimal_k(X_scaled)
plot_elbow(inertias)
plot_silhouette(silhouettes)


# 3. Ejecutar K-Means con K=5
print("Ejecutando K-Means...")
labels_km, inertia, sil_km, model_km = run_kmeans(X_scaled, k=5)
plot_clusters(X_scaled, labels_km, "K-Means (K=5)")
print("Silhouette K-Means:", sil_km)
print("Evaluación:", evaluate_clustering(X_scaled, labels_km))


# 4. Clustering Jerárquico
print("Ejecutando Clustering Jerárquico...")
labels_hc, sil_hc, model_hc = run_hierarchical(X_scaled, n_clusters=5)
plot_clusters(X_scaled, labels_hc, "Clustering Jerárquico")
print("Silhouette HC:", sil_hc)
print("Evaluación:", evaluate_clustering(X_scaled, labels_hc))


# 5. DBSCAN
print("Ejecutando DBSCAN...")
labels_db, sil_db, model_db = run_dbscan(X_scaled, eps=0.5, min_samples=5)
plot_clusters(X_scaled, labels_db, "DBSCAN")
print("Silhouette DBSCAN:", sil_db)
print("Evaluación:", evaluate_clustering(X_scaled, labels_db))