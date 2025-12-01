from sklearn.cluster import KMeans

def train_kmeans(X_train, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X_train)
    return model
