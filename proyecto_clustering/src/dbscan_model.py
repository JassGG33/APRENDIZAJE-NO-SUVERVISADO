from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np




def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)


    # DBSCAN puede crear ruido (-1), evitar error
    if len(set(labels)) > 1 and -1 not in set(labels):
        silhouette = silhouette_score(X_scaled, labels)
    else:
        silhouette = "No válido (ruido o 1 clúster)"

    return labels, silhouette, model