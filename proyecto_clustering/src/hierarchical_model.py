from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def run_hierarchical(X_scaled, n_clusters=5, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    return labels, silhouette, model