import pandas as pd
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
    sns.pairplot(df, hue=house_column, palette=house_color, height=1.5)  # height ajuste chaque sous-graphique
    
    # plt.gcf().set_size_inches(14, 14)

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
    df = preprocesing(df)

    # course_columns = extract_course_columns(df)

    plot_pair_plot(df)