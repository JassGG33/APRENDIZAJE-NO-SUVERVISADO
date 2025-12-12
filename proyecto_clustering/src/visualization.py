import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters(X_scaled, labels, title):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap="viridis")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def plot_elbow(inertias):
    plt.plot(range(2, len(inertias) + 2), inertias, marker="o")
    plt.title("Método del Codo (K-Means)")
    plt.xlabel("K")
    plt.ylabel("Inercia")
    plt.show()

def plot_silhouette(scores):
    plt.plot(range(2, len(scores) + 2), scores, marker="o")
    plt.title("Silhouette por K")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.show()