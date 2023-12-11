from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

centers_part1 = [[0.5, 2], [-1, -1], [1.5, -1]]
X_part1, y_part1 = make_blobs(n_samples=200, centers=centers_part1, cluster_std=0.500, random_state=0)
X_part1 = StandardScaler().fit_transform(X_part1)

plt.figure(figsize=(7, 6))
plt.scatter(X_part1[:, 0], X_part1[:, 1], c=y_part1, cmap='Paired')
plt.show()

db_part1 = DBSCAN(eps=0.4, min_samples=20)
db_part1.fit(X_part1)
y_pred_part1 = db_part1.fit_predict(X_part1)

plt.figure(figsize=(10, 6))
plt.scatter(X_part1[:, 0], X_part1[:, 1], c=y_pred_part1, cmap='Paired')
plt.title("Clusters determined by DBSCAN (Dataset 1)")
plt.show()

labels_part1 = db_part1.labels_
labels_count_part1 = list(labels_part1).count(-1)
print('Counted number of points classified as noise : %d' % labels_count_part1)


dd_part2 = pd.read_csv('wine_data.csv')
DD_part2 = pd.DataFrame(dd_part2)

DD_cluster_part2 = DD_part2[['Alcohol', 'Ash']]

plt.figure(figsize=(10, 6))
plt.scatter(DD_part2['Alcohol'], DD_part2['Ash'])
plt.title("Scatter plot of the second dataset")
plt.show()

db_part2 = DBSCAN(eps=0.3, min_samples=3)
y_pred_part2 = db_part2.fit_predict(DD_cluster_part2)

plt.figure(figsize=(10, 6))
plt.scatter(DD_cluster_part2['Alcohol'], DD_cluster_part2['Ash'], c=y_pred_part2, cmap='Paired')
plt.title("Кластеризація визначена DBSCAN (Набір даних 2)")
plt.show()

labels_noise_count_part2 = db_part2.labels_[db_part2.labels_ == -1].size
total_points_count_part2 = db_part2.labels_[db_part2.labels_].size
print('Кількість точок, класифікованих як шум: %d' % labels_noise_count_part2)
print('Загальна кількість точок : %d' % total_points_count_part2)
