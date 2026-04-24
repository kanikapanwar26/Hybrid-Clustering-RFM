import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

data = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')

data = data.dropna()
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

today = data['InvoiceDate'].max()

rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (today - x.max()).days,
    'InvoiceNo': 'count',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency','Frequency','Monetary']
rfm = rfm.reset_index()


X = rfm[['Recency','Frequency','Monetary']].values
X = (X - X.mean(axis=0)) / X.std(axis=0)

def kmeans(X, k=3, max_iter=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels

rfm['KMeans'] = kmeans(X)

def dbscan(X, eps=0.8, min_samples=5):
    labels = np.full(len(X), -1)
    cluster_id = 0

    for i in range(len(X)):
        if labels[i] != -1:
            continue

        neighbors = np.where(np.linalg.norm(X - X[i], axis=1) < eps)[0]

        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[neighbors] = cluster_id
            cluster_id += 1

    return labels

rfm['DBSCAN'] = dbscan(X)

def fuzzy_cmeans(X, k=3, max_iter=100):
    n = len(X)
    np.random.seed(42)

    U = np.random.rand(n, k)
    U = U / np.sum(U, axis=1, keepdims=True)

    for _ in range(max_iter):
        centers = (U.T @ X) / np.sum(U.T, axis=1)[:, None]

        dist = np.linalg.norm(X[:, None] - centers, axis=2)
        dist = np.fmax(dist, 1e-6)

        U = 1 / dist
        U = U / np.sum(U, axis=1, keepdims=True)

    return np.argmax(U, axis=1)

rfm['FCM'] = fuzzy_cmeans(X)


rfm['Hybrid_KM_FCM'] = (rfm['KMeans'] + rfm['FCM']) // 2
rfm['Hybrid_FCM_DB'] = (rfm['FCM'] + rfm['DBSCAN']) // 2
rfm['Hybrid_KM_DB'] = (rfm['KMeans'] + rfm['DBSCAN']) // 2
rfm['Hybrid_ALL'] = (rfm['KMeans'] + rfm['FCM'] + rfm['DBSCAN']) // 3

def compute_silhouette(X, labels, name):
    # remove noise (-1)
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        score = silhouette_score(X[mask], labels[mask])
        print(f"{name} Silhouette Score: {score:.4f}")
    else:
        print(f"{name} Silhouette Score: Not valid (only 1 cluster or noise)")

compute_silhouette(X, rfm['KMeans'].values, "KMeans")
compute_silhouette(X, rfm['DBSCAN'].values, "DBSCAN")
compute_silhouette(X, rfm['FCM'].values, "FCM")

compute_silhouette(X, rfm['Hybrid_KM_FCM'].values, "Hybrid KM+FCM")
compute_silhouette(X, rfm['Hybrid_FCM_DB'].values, "Hybrid FCM+DBSCAN")
compute_silhouette(X, rfm['Hybrid_KM_DB'].values, "Hybrid KM+DBSCAN")
compute_silhouette(X, rfm['Hybrid_ALL'].values, "Hybrid ALL")

def plot(data, labels, title):
    plt.figure()
    scatter = plt.scatter(data['Frequency'], data['Monetary'], c=labels)
    plt.xlabel("Frequency")
    plt.ylabel("Monetary")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True)
    plt.show()

plot(rfm, rfm['KMeans'], "K-Means")
plot(rfm, rfm['DBSCAN'], "DBSCAN")
plot(rfm, rfm['FCM'], "Fuzzy C-Means")

plot(rfm, rfm['Hybrid_KM_FCM'], "Hybrid (KMeans + FCM)")
plot(rfm, rfm['Hybrid_FCM_DB'], "Hybrid (FCM + DBSCAN)")
plot(rfm, rfm['Hybrid_KM_DB'], "Hybrid (KMeans + DBSCAN)")
plot(rfm, rfm['Hybrid_ALL'], "Hybrid (All Combined)")