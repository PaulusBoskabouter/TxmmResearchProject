import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import json
from time import sleep







def plot_elbow_method(vectors, max_clusters=20):
        distortions = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(vectors)
            distortions.append(kmeans.inertia_)  # Sum of squared distances to cluster centers
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, max_clusters + 1), distortions, marker='o', linestyle='--')
        plt.title('Elbow Method to Determine Optimal Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion (Inertia)')
        plt.show()



if __name__ == "__main__":
    root_path = ".\\dataset\\"
    stop_words = set(stopwords.words('dutch'))

    year_data = {}
    document_matrix = []
    for year in range(1850, 1851):
        print(f"Opening {year:4}")
        with open(f"{year}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
            for article in data.keys():
                del data[article]["paragraphs"] # Oops not enough RAM, we'll bypass this for now.
                for paragraph in data[article].get("embedded"):
                    try:
                        print(len(paragraph))
                        document_matrix.append(np.array(paragraph))
                    except TypeError:
                        pass
                #data[article]["embedded"] = np.mean(np.array(data[article]["embedded"]), axis=0)
                #document_vector_means.append(data[article]["embedded"])
                     
            year_data[year] = data




    
    a = np.array(document_matrix)
    
    print(a.shape)
    


    plot_elbow_method(a, max_clusters=30)

    # # 2. Choose the optimal number of clusters (from the elbow plot, for example k=3)
    optimal_clusters = 9
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(document_matrix)

    # 3. Assign clusters to documents
    document_clusters = kmeans.labels_

    # 4. Evaluate the clustering (optional)
    silhouette_avg = silhouette_score(document_matrix, document_clusters)
    print(f"Silhouette Score: {silhouette_avg}")

    # 5. Visualize the Clusters (Optional: Using PCA for dimensionality reduction)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(document_matrix)

    plt.figure(figsize=(8, 6))
    for cluster in range(optimal_clusters):
        cluster_points = reduced_vectors[document_clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    plt.title('K-Means Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()