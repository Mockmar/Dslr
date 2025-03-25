from preprocessing import Preprocess, CustomStandardScaler, CustomLabelEncoder
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
import joblib

def import_csv(path):
    return pd.read_csv(path)

class Predictor:
    def __init__(self, path_model):
        """
        Initialise le modèle avec un taux d'apprentissage et un nombre d'itérations pour la descente de gradient.
        """
        self.path_model = path_model
        self.thetas = []
        self.classes_ = []

    def parse_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    theta_tmp = []
                    nom_class, thetas = line.split(":")
                    self.classes_.append(nom_class)
                    theta_tmp = list(map(float, thetas.split(",")))
                    self.thetas.append(theta_tmp)
        self.thetas = np.array(self.thetas)

    def sigmoid(self, z):
        """
        Fonction sigmoïde : transforme n'importe quelle valeur en un score entre 0 et 1.
        """
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        """
        Retourne les probabilités pour chaque classe.
        """
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_bias @ self.thetas.T)

    def predict(self, X):
        """
        Prédit la classe en prenant celle avec la plus grande probabilité.
        """
        probabilities = self.predict_proba(X)
        encoded_classes = np.argmax(probabilities, axis=1)
        result = [self.classes_[i] for i in encoded_classes]
        return result

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide a path to the dataset and a path to the model')
        sys.exit(1)
    path = sys.argv[1]
    path_model = sys.argv[2]
    df = import_csv(path)

    
    feature_name = ['Astronomy',
                    'Herbology',
                    'Defense Against the Dark Arts',
                    'Divination',
                    'Muggle Studies',
                    'Ancient Runes',
                    'History of Magic',
                    'Transfiguration',
                    'Charms',
                    'Flying']
    
    columns = feature_name
    columns.append('Index')

    print(df.columns)
    print(columns)

    df = df[columns]
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)

    index_list = df['Index'].tolist()
    df.drop(columns=['Index'], inplace=True)

    normalize = CustomStandardScaler()
    normalize.load('normalizer.csv')

    features = normalize.transform(df)
    X = features.to_numpy()

    model = Predictor(path_model=path_model)
    model.parse_file(model.path_model)
    predict = model.predict(X)
    with open('houses.csv', 'w') as file:
        file.write(f'Index,Hogwarts House\n')
        for index, p in zip(index_list, predict):
            file.write(f'{index},{p}\n')
