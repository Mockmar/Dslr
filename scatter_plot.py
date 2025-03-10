import pandas as pd
import matplotlib.pyplot as plt
import sys

def import_csv(path):
    return pd.read_csv(path)

def find_most_similar_features(df):
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()
    max_corr = 0
    feature_pair = (None, None)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                feature_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])
    return feature_pair

def plot_scatter(df, feature1, feature2):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature1], df[feature2], alpha=0.5)
    plt.title(f'Scatter Plot between {feature1} and {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please provide a path to the dataset')
        sys.exit(1)
    path = sys.argv[1]
    df = import_csv(path)
    
    # Trouver les deux caractéristiques les plus similaires
    feature1, feature2 = find_most_similar_features(df)
    
    # Afficher le nuage de points pour les deux caractéristiques les plus similaires
    plot_scatter(df, feature1, feature2)