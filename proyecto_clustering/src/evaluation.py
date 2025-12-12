from sklearn.metrics import davies_bouldin_score

def evaluate_clustering(X_scaled, labels):
    if len(set(labels)) < 2:
        return {
            "Davies-Bouldin": "No válido",
        }


    return {
        "Davies-Bouldin": davies_bouldin_score(X_scaled, labels)
    }