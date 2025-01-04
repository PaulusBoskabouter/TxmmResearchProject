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





def plot_elbow_method(vectors, max_clusters):
    distortions = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(vectors)
        distortions.append(kmeans.inertia_)  # Sum of squared distances to cluster centers
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), distortions, marker='o', linestyle='--')
    # Increase the font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Elbow Method to Determine Optimal Clusters', fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Distortion (Inertia)', fontsize=14)
    plt.savefig('Elbow_plot.png')
    print("Saved Elbowplot!")


def plot_kmeans_cluster(document_matrix, optimal_clusters=5):
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(document_matrix)

    # 3. Assign clusters to documents
    cluster_labels = kmeans.labels_

    # 4. Evaluate the clustering (optional)
    silhouette_avg = silhouette_score(document_matrix, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # 5. Visualize the Clusters (Optional: Using PCA for dimensionality reduction)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(document_matrix)

    plt.figure(figsize=(8, 6))
    for cluster in range(optimal_clusters):
        cluster_points = reduced_vectors[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster+1}', alpha=.8)
    plt.title('K-Means Clustering Visualization', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    plt.legend()
    plt.savefig(f"K-means cluster.png")
    print("Saved Clusterplot!")
    return cluster_labels



def create_wordclouds(optimal_clusters):
    year_to_cluster = {}

    for y in years:
        year_to_cluster[y] = [0 for _ in range(optimal_clusters)]
    
    for target_cluster in range(optimal_clusters):
        print(f"Creating wordcloud for cluster {target_cluster}")
        index = 0
        words = []
        for year in years:
            print(f"Opening {year:4}", end='\r')
            i = 0
            with open(f"preprocessed\\{year}.json", 'r', encoding='utf-8') as file:
                data = json.load(file)
                for paragraph in data.get("paragraphs"):
                    if cluster_labels[index] == target_cluster:
                        year_to_cluster[year][target_cluster] += 1
                        valid_tokens = [word.lower() for word in paragraph if word.isalpha()]  # Keep only alphabetic words
                        words += valid_tokens
                    i += 1
                    index += 1

        
        word_counts = Counter(words)
        # Generate the Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

        # Plot the Word Cloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")  # Turn off the axis
        plt.title(f"Word Cloud of cluster {target_cluster+1}", fontsize=16)
        plt.savefig(f"Wordcloud_cluster_{target_cluster+1}.png")
        print(f"Saved Wordcloud for cluster {target_cluster+1}")
    return year_to_cluster
    
def create_stacked_barplot(year_to_cluster):
    # Convert data to a NumPy array for easier manipulation
    years = list(year_to_cluster.keys())
    clusters = len(next(iter(year_to_cluster.values())))  # Number of clusters, based on first entry
    article_counts = np.array([year_to_cluster[year] for year in years])

    # Grouping by 20 years
    group_by_years = 20
    grouped_years = []
    grouped_counts = []


    for i in range(0, len(years)-1, group_by_years):
        # Group the years together and average the counts for each cluster
        grouped_years.append(f"{years[i]}-{years[i] + group_by_years}")
        avg_counts = np.mean(article_counts[i:i+group_by_years], axis=0)
        grouped_counts.append(avg_counts)

    # Convert the grouped counts to a numpy array for easier handling
    grouped_counts = np.array(grouped_counts)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create stacked bar chart
    ax.bar(grouped_years, grouped_counts[:, 0], label=f'Cluster 1')
    for i in range(1, clusters):
        ax.bar(grouped_years, grouped_counts[:, i], bottom=np.sum(grouped_counts[:, :i], axis=1), label=f'Cluster {i+1}')

    # Add labels and title
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Year Range', fontsize=14)
    ax.set_ylabel('Number of Articles', fontsize=14)
    ax.set_title('Number of Articles in Each Cluster (Grouped by 29 Years)', fontsize=16)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()  # Ensure everything fits
    plt.savefig("Stacked_barchart_cluster_division.png")
    print("Saved barchart!")



if __name__ == "__main__":
    root_path = ".\\dataset\\"
    stop_words = set(stopwords.words('dutch'))

    years = range(1855, 1996)

    # 0. Load the data (but only the embedded paragraphs) as saving the words uses too much memory.
    document_matrix = []
    num_paragrahps = 0
    used_paragraphs = 0
    print(f"Starting k-means clustering")
    for year in years:
        print(f"Opening {year:4}", end='\r')
        with open(f"preprocessed\\{year}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
            i = 0
            num_paragrahps += data.get("total_articles")
            used_paragraphs += data.get("used_articles")
            for paragraph in data.get("embedded"):
                document_matrix.append(np.array(paragraph))
                i += 1
    
    print("total paragraphs: ", num_paragrahps)
    print("used paragraphs: ", used_paragraphs)
            
    document_matrix = np.array(document_matrix)
    
    print(document_matrix.shape)
    
    # 1. Determine the optimal number of clusters
    plot_elbow_method(document_matrix, max_clusters=15)


    # 2. Create the KMeans cluster plot
    cluster_labels = plot_kmeans_cluster(document_matrix, optimal_clusters=6)
    

    # 3. Create wordclouds for each cluster
    year_to_cluster = create_wordclouds(optimal_clusters=6)
    

    # 4. Create a bar plot of the number of paragraphs in each cluster for each year
    create_stacked_barplot(year_to_cluster)