import pandas as pd
import matplotlib.pyplot as plt
import sys

def import_csv(path):
    return pd.read_csv(path)

def plot_histograms(df, course_columns, house_column='Hogwarts House'):
    # houses = df[house_column].unique()
    # houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    houses = {'Gryffindor': 'red', 'Slytherin': 'green', 'Ravenclaw': 'blue', 'Hufflepuff': 'yellow'}
    for course in course_columns:
        plt.figure(figsize=(10, 6))
        for house, color in houses.items():
            subset = df[df[house_column] == house]
            plt.hist(subset[course], bins=20, alpha=0.5, label=house, color=color)
        plt.title(f'Distribution des scores pour {course}')
        plt.xlabel('Score')
        plt.ylabel('Fréquence')
        plt.legend(loc='upper right')
        plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please provide a path to the dataset')
        sys.exit(1)
    path = sys.argv[1]
    df = import_csv(path)
    # Remplacez 'course_columns' par les noms des colonnes de vos cours
    course_columns = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    
    plot_histograms(df, course_columns)