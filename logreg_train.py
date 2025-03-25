from preprocessing import CustomStandardScaler
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def import_csv(path):
    return pd.read_csv(path)

class LogisticRegressionOVR:
    def __init__(self, learning_rate=0.01, max_iter=1000, class_encoder=None):
        """
        Initialise le modèle avec un taux d'apprentissage et un nombre d'itérations pour la descente de gradient.
        """
        self.claas_encoder = class_encoder or {}
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.thetas = []  # Liste pour stocker les poids de chaque modèle
        self.classes_ = None  # Stocker les noms des classes
        self.log_loss = []

    def sigmoid(self, z):
        """
        Fonction sigmoïde : transforme n'importe quelle valeur en un score entre 0 et 1.
        """
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, theta):
        """
        Fonction de coût : mesure l'erreur entre les prédictions et les vraies valeurs.
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def gradient_descent(self, X, y, theta, class_name):
        """
        Applique la descente de gradient pour trouver les poids optimaux.
        """
        m = len(y)
        log_loss_class = []
        for _ in tqdm(range(self.max_iter), desc=f"Training model for {class_name}", leave=True):
            h = self.sigmoid(X @ theta)
            gradient = (1 / m) * X.T @ (h - y)  # Gradient de la fonction de coût
            theta -= self.learning_rate * gradient  # Mise à jour des poids
            log_loss_class.append(self.cost_function(X, y, theta))
        self.log_loss.append(log_loss_class)
        return theta

    def fit(self, X, y):
        """
        Entraîne le modèle en utilisant la méthode One-vs-All.
        """
        m, n = X.shape
        self.classes_ = np.unique(y)
        self.thetas = np.zeros((len(self.classes_), n + 1))
        X_bias = np.c_[np.ones((m, 1)), X]

        for i, c in enumerate(self.classes_):
            y_binary = (y == c).astype(int)  # Convertir en problème binaire (1 si c'est la classe, sinon 0)
            theta = np.zeros(n + 1)  # Initialiser les poids
            self.thetas[i] = self.gradient_descent(X_bias, y_binary, theta, c)  # Entraîner le modèle

    def predict_proba(self, X):
        """
        Retourne les probabilités pour chaque classe.
        """
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Ajouter le biais
        return self.sigmoid(X_bias @ self.thetas.T)  # Matrice de probabilités

    def predict(self, X):
        """
        Prédit la classe en prenant celle avec la plus grande probabilité.
        """
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]  # Retourne la classe avec la proba max

    def plot_log_loss(self):
        """
        Affiche la courbe de la fonction de coût pour chaque classe.
        """
        for i, c in enumerate(self.classes_):
            plt.plot(self.log_loss[i], label=c)
        plt.title('Log Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()

    def export_model(self, path):
        """
        Exporte le modèle pour une utilisation ultérieure.
        """
        with open(path, 'w') as f:
            for i, theta in enumerate(self.thetas):
                f.write(f'{self.classes_[i]}:{",".join(map(str, theta))}\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide a path to the dataset')
        sys.exit(1)
    path = sys.argv[1]
    df = import_csv(path)
    df.drop(columns=['Index'], inplace=True)

    normalize = CustomStandardScaler()
    model = LogisticRegressionOVR(learning_rate=0.1, max_iter=10000)

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

    target_name = ['Hogwarts House']

    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)

    features = df[feature_name]
    target = df[target_name]

    features = normalize.fit_transform(features)
    normalize.save('normalizer.csv')

    X = features.to_numpy()
    y = target.to_numpy().ravel()


    model.fit(X, y)
    model.export_model('model.txt')
    model.plot_log_loss()
