import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, all_theta):
    X = np.c_[np.ones(X.shape[0]), X]  # Ajouter biais
    probs = sigmoid(X @ all_theta.T)  # Calcul des probabilités
    return np.argmax(probs, axis=1)  # Classe avec la probabilité max

if __name__ == "__main__":
    # Charger les données de test
    X_test = pd.read_csv("dataset_test.csv").values
    
    # Charger les poids entraînés
    all_theta = np.loadtxt("weights.csv", delimiter=",")

    # Faire les prédictions
    predictions = predict(X_test, all_theta)

    # Sauvegarder dans houses.csv
    pd.DataFrame(predictions, columns=["Predicted"]).to_csv("houses.csv", index=False)
    print("✅ Prédictions terminées. Résultats sauvegardés dans houses.csv")
