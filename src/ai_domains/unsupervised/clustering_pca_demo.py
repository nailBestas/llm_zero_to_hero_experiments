from __future__ import annotations

import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def run_clustering(n_clusters: int = 3, random_state: int = 42):
    data = load_iris()
    X = data.data  # [150, 4]
    y_true = data.target  # 0,1,2

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    ari = adjusted_rand_score(y_true, cluster_labels)
    sil = silhouette_score(X_scaled, cluster_labels)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("=== Unsupervised Learning: K-Means + PCA on Iris ===")
    print(f"n_clusters          : {n_clusters}")
    print(f"Adjusted Rand Index : {ari:.4f}")
    print(f"Silhouette score    : {sil:.4f}")
    print("\nİlk 5 örneğin (PCA 2D) koordinatları ve cluster etiketleri:")
    for i in range(5):
        x0, x1 = X_pca[i]
        print(
            f"Sample {i:3d}: PCA=({x0: .3f}, {x1: .3f}), "
            f"true_label={y_true[i]}, cluster={cluster_labels[i]}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_clustering(n_clusters=args.clusters, random_state=args.seed)


if __name__ == "__main__":
    main()
