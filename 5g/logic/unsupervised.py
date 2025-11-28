
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_kmeans(df, n_clusters=3):
    df = df.dropna()  #Drop rows with NaNs

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["Cluster"] = clusters
    return df_clustered

def run_pca(df, n_components=2):
    df = df.select_dtypes(include='number')  # only numeric columns
    df = df.dropna()  #Drop rows with NaNs

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=components, columns=[f"PC{i+1}" for i in range(n_components)])
    return pca_df, pca.explained_variance_ratio_

