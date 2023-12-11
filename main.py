from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd



def plot_scatter(X, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(title)
    plt.show()


def plot_dbscan_result(X, y_pred, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
    plt.title(title)
    plt.show()


def run_dbscan(X, eps, min_samples, title):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = db.fit_predict(X)
    plot_dbscan_result(X, y_pred, title)

    labels_noise_count = db.labels_[db.labels_ == -1].size
    total_points_count = db.labels_[db.labels_].size
    print('Кількість точок, класифікованих як шум: %d' % labels_noise_count)
    print('Загальна кількість точок: %d' % total_points_count)


# Задані параметри для make_blobs
n_samples_part1 = 200
cluster_std_part1 = 0.3

X_part1, y_part1 = make_blobs(n_samples=n_samples_part1, centers=1, cluster_std=cluster_std_part1, random_state=42)
X_part1 = StandardScaler().fit_transform(X_part1)
plot_scatter(X_part1, "Синтетичні дані для DBSCAN (make_blobs)")
run_dbscan(X_part1, eps=0.5, min_samples=2, title="Кластеризація за допомогою DBSCAN (make_blobs)")

wine_data = pd.read_csv('wine_data.csv')
DD_cluster_part2 = wine_data[['Alcohol', 'Ash']]
plot_scatter(DD_cluster_part2.values, "Діаграма розсіювання для wine_data.csv")
run_dbscan(DD_cluster_part2.values, eps=0.3, min_samples=3, title="Кластеризація за допомогою DBSCAN для wine_data.csv")
