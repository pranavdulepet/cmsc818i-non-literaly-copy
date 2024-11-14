# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_by_category(results_path='/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results.csv'):
    results_df = pd.read_csv(results_path)

    avg_similarities = results_df.groupby('category')[['levenshtein_similarity', 'cosine_similarity']].mean().reset_index()

    # Plot Levenshtein Similarity
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_similarities, x='category', y='levenshtein_similarity')
    plt.title('Average Levenshtein Similarity by Category')
    plt.ylabel('Levenshtein Similarity')
    plt.xlabel('Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Cosine Similarity
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_similarities, x='category', y='cosine_similarity')
    plt.title('Average Cosine Similarity by Category')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_similarity_by_category()
