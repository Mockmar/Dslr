import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        theta -= (alpha/m) * (X.T @ (h - y))
    return theta

def train_logreg_one_vs_all(X, y, num_classes, alpha=0.1, num_iters=1000):
    m, n = X.shape
    X = np.c_[np.ones(m), X]  # Ajouter un biais (colonne de 1)
    all_theta = np.zeros((num_classes, n + 1))

    for i in range(num_classes):
        y_binary = (y == i).astype(int)  # One-vs-All : chaque classe est traitée comme binaire
        theta = np.zeros(n + 1)
        all_theta[i] = gradient_descent(X, y_binary, theta, alpha, num_iters)
    
    return all_theta

if __name__ == "__main__":
    # Charger les données
    data = pd.read_csv("datasets/dataset_train.csv")
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels (classes)
    
    num_classes = len(np.unique(y))  # Nombre de classes différentes
    all_theta = train_logreg_one_vs_all(X, y, num_classes)

    # Sauvegarder les poids entraînés
    np.savetxt("weights.csv", all_theta, delimiter=",")
    print("✅ Entraînement terminé. Poids sauvegardés dans weights.csv")
