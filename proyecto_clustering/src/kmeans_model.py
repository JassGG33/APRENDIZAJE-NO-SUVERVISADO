from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


def run_kmeans(X_scaled, k=5):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    return labels, inertia, silhouette, model


def find_optimal_k(X_scaled, max_k=10):
    inertias = []
    silhouettes = []


    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        
    return inertias, silhouettes