import pandas as pd
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import Preprocess
import sys

def import_csv(path):
    return pd.read_csv(path)

def preprocesing(df):
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['Index'], inplace=True)
    return df

def plot_pair_plot(df, house_column='Hogwarts House'):
    house_color = {
        'Ravenclaw': '#80bfff',  # Bleu pastel
        'Slytherin': '#85e085',  # Vert pastel
        'Gryffindor': '#ff9999',  # Rouge pastel
        'Hufflepuff': '#ffeb99'   # Jaune pastel
    }
    g = sns.pairplot(df, hue=house_column, palette=house_color)

    for ax in g.axes.flatten():
        # rotate x axis labels
        # ax.set_xlabel(ax.get_xlabel(), rotation = 90)
        # rotate y axis labels
        ax.set_ylabel(ax.get_ylabel(), rotation = 0)
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    
    plt.show(block=True)

def extract_course_columns(df):
    return df.select_dtypes(include=['float64', 'int64']).columns.tolist()

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

    plot_pair_plot(df)