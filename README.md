# 🧠 Machine Learning Playground with Streamlit
Lien streamlit APP : https://projetdigimlstreamilt-c3ybvsxevneutw4fxukoou.streamlit.app/
Bienvenue dans le **Machine Learning Playground**, une application interactive développée avec Streamlit. Ce projet propose un espace pour expérimenter avec des modèles de classification, de régression, et de détection d'ongles à partir d'images, le tout en utilisant une interface simple et intuitive. C'est un projet collaboratif avec une équipe de passionnés de machine learning et d'intelligence artificielle.

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Contribuer](#-contribuer)
- [Licence](#-licence)

## 🎯 Fonctionnalités

- **Page d'accueil** : Introduction au projet et à son objectif.
- **Équipe** : Présentation des membres qui ont collaboré à ce projet.
- **Classification** : Implémentez et évaluez des modèles de classification sur des données personnalisées.
- **Régression** : Entraînez des modèles de régression pour prédire des valeurs continues.
- **Détection d'ongles** : Modèle de détection d'images pour identifier les ongles dans des photos.
- **Conclusion** : Résumé des résultats obtenus avec des suggestions pour de futures améliorations.

## 🚀 Installation

### Prérequis

- [Python 3.7+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/) (gestionnaire de paquets Python)

### Étapes d'installation

1. Clonez ce dépôt GitHub dans votre machine locale :

    ```bash
    git clone https://github.com/votre-utilisateur/machine-learning-playground.git
    ```

2. Accédez au répertoire du projet :

    ```bash
    cd machine-learning-playground
    ```

3. Installez les dépendances nécessaires :

    ```bash
    pip install -r requirements.txt
    ```

4. Lancez l'application Streamlit :

    ```bash
    streamlit run app.py
    ```

5. Ouvrez un navigateur et allez à l'adresse suivante (si elle ne s'ouvre pas automatiquement) :

    ```
    http://localhost:8501
    ```

## 🛠 Utilisation

### 1. Page d'accueil
- La page principale du projet, qui vous donne un aperçu général des fonctionnalités et objectifs.

### 2. Équipe
- Consultez cette page pour découvrir les membres de l'équipe qui ont contribué à ce projet.

### 3. Classification
- Naviguez vers la page "Classification" pour importer vos datasets et tester différents modèles de classification. Vous pourrez également évaluer les performances à l’aide de matrices de confusion et de scores de précision.

### 4. Régression
- Utilisez la page "Régression" pour prédire des valeurs continues. Cette section vous permet d'explorer les modèles de régression linéaire, de visualiser les prédictions et de comparer les résultats.

### 5. Détection d'ongles
- Téléchargez des images et utilisez des modèles de détection pour identifier automatiquement des ongles dans les photos. Idéal pour des applications dans le domaine de la beauté et des soins personnels.

### 6. Conclusion
- Un résumé des résultats de vos modèles, ainsi que des suggestions pour améliorer les performances ou essayer de nouvelles approches.

## 📂 Structure du projet

```plaintext
my_ml_playground/
│
├── pages/                # Contient les fichiers des pages
│   ├── 1_Home.py         # Page d'accueil
│   ├── 2_Team.py         # Page présentant l'équipe
│   ├── 3_Classification.py  # Page pour la classification
│   ├── 4_Regression.py      # Page pour la régression
│   ├── 5_Nail_Detection.py  # Page pour la détection d'ongles
│   ├── 6_Conclusion.py      # Page de conclusion
│
├── data/               # Contient les ressources (datasets, images, etc.)
│   ├── diabete.csv/           # Jeux de données diabete
│   ├── vin.csv/               # Jeux de données vin
│
├── img/               # Contient les ressources (images, etc.)
│   ├── images.png/           
│
├── models/               # Modèles sauvegardés après entraînement
│   ├── saved_model.pkl   # Modèle sauvegardé
│
├── README.md             # Documentation du projet
├── requirements.txt      # Fichier des dépendances
└── app.py                # Fichier principal de l'application Streamlit
```

## 🤝 Contribuer

Les contributions sont les bienvenues ! Si vous avez des idées pour améliorer ce projet, suivez les étapes ci-dessous :

1. Forkez le projet.
2. Créez une branche pour vos modifications (`git checkout -b feature/ma-nouvelle-fonctionnalité`).
3. Committez vos changements (`git commit -m 'Ajout d'une nouvelle fonctionnalité'`).
4. Poussez vers votre branche (`git push origin feature/ma-nouvelle-fonctionnalité`).
5. Ouvrez une Pull Request.

## 📝 Licence

Ce projet est sous licence MIT. Vous pouvez voir plus de détails dans le fichier [LICENSE](./LICENSE).

