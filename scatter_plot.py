import pandas as pd
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from preprocessing import Preprocess

def import_csv(path):
    return pd.read_csv(path)

def preprocesing(df):
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['Index'], inplace=True)
    return df

def correlation_matrix_extraction(df):
    numereic_df = df.select_dtypes(include=['float64', 'int64'])
    return numereic_df.corr().abs()

def find_most_similar_features(correlation_matrix):
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
    plt.figure(figsize=(14, 8))
    plt.scatter(df[feature1], df[feature2], alpha=0.5)
    plt.title(f'Scatter Plot between {feature1} and {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show(block=True)

def plot_matrix_correlation(correlation_matrix):
    plt.figure(figsize=(14, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.subplots_adjust(left=0.1, right=1, top=0.95, bottom=0.2, wspace=0.3, hspace=0.3)
    plt.show(block=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please provide a path to the dataset')
        sys.exit(1)
    path = sys.argv[1]

    df = import_csv(path)

    preprocess = Preprocess()
    df = preprocess.preprocesing(df)
    df = preprocess.calculate_age(df, 2020)
    df = preprocess.encode_column(df, 'Best Hand')

    correlation_matrix = correlation_matrix_extraction(df)

    plot_matrix_correlation(correlation_matrix)

    feature1, feature2 = find_most_similar_features(correlation_matrix)
    
    plot_scatter(df, feature1, feature2)