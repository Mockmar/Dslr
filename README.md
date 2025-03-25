# Dslr

## Description
Ce projet est un pipeline d'analyse de données et de machine learning pour prédire les affectations des maisons de Poudlard en fonction de diverses caractéristiques des étudiants. Il comprend la prétraitement des données, l'analyse statistique et les outils de visualisation, ainsi qu'un modèle de régression logistique pour la classification.

## Structure du projet
- `stats.py`: Fonctions statistiques personnalisées pour l'analyse des données.
- `scatter_plot.py`: Script pour générer des diagrammes de dispersion et des matrices de corrélation.
- `preprocessing.py`: Utilitaires de prétraitement des données, y compris l'encodage des labels et la normalisation standard.
- `pair_plot.py`: Script pour générer des diagrammes en paires pour visualiser les relations entre les caractéristiques.
- `logreg_train.py`: Script pour entraîner un modèle de régression logistique en utilisant la stratégie One-vs-Rest.
- `logreg_predict.py`: Script pour prédire les affectations des maisons de Poudlard en utilisant un modèle de régression logistique entraîné.
- `histogram.py`: Script pour générer des histogrammes des distributions des caractéristiques.
- `describe.py`: Script pour générer des statistiques descriptives pour le jeu de données.
- `.gitignore`: Fichier gitignore pour exclure les fichiers inutiles du contrôle de version.

## Installation
1. Cloner le dépôt :
    ```sh
    git clone <repository_url>
    cd Dslr
    ```
2. Installer les packages Python requis :
    ```sh
    pip install -r requirements.txt
    ```

## Utilisation

### Prétraitement des données
Pour prétraiter le jeu de données :
```sh
python preprocessing.py <path_to_dataset>
```

### Entraînement du modèle
Pour entraîner le modèle de régression logistique :
```sh
python logreg_train.py <path_to_dataset>
```

### Faire des prédictions
Pour faire des prédictions en utilisant le modèle entraîné :
```sh
python logreg_predict.py <path_to_dataset> <path_to_model>
```

### Génération de visualisations
Pour générer des diagrammes de dispersion et des matrices de corrélation :
```sh
python scatter_plot.py <path_to_dataset>
```

Pour générer des diagrammes en paires :
```sh
python pair_plot.py <path_to_dataset>
```

Pour générer des histogrammes :
```sh
python histogram.py <path_to_dataset>
```

### Statistiques descriptives
Pour générer des statistiques descriptives pour le jeu de données :
```sh
python describe.py <path_to_dataset>
```

## Licence
Ce projet est sous licence MIT.
