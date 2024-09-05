import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Titre de la sous-page
st.title("Analyse de Données : Exploration et analyses visuelles des données pour la Régression")

# Phrase explicative d'en-tête
st.markdown("""
Bienvenue sur la page d'analyse des données. Cette section vous permet d'explorer, visualiser et comprendre 
les caractéristiques du jeu de données avant la modélisation. L'objectif est de mettre en évidence les relations entre les variables, 
d'identifier les tendances et les anomalies, et de préparer les données pour les étapes de régression.

#### **Fonctionnalités Disponibles :**
- **Résumé Statistique** : Explorez les statistiques descriptives (moyenne, écart-type, etc.) pour chaque variable.
- **Visualisations Interactives** : Créez des graphiques pour visualiser les distributions des variables, les corrélations, et les relations importantes entre les caractéristiques et la variable cible.
- **Sélection de Variables** : Filtrez et choisissez les variables pertinentes pour améliorer les performances des modèles de régression.
- **Gestion des Données Manquantes** : Identifiez et gérez les données manquantes ou aberrantes pour garantir une analyse robuste.
- **Transformation des Données** : Testez différentes transformations (normalisation, standardisation) pour améliorer les modèles.
""")

# Fonction pour analyse descriptive

def viz_analyse_descriptive(X):
    st.subheader("Analyse Descriptive")
    variable = st.selectbox("Choisissez une variable à analyser :", options=list(X.columns))

    st.write(f"Histogramme de {variable}")
    fig, ax = plt.subplots()
    sns.histplot(X[variable], kde=True, ax=ax)
    st.pyplot(fig)

    if st.checkbox("Afficher le boxplot"):
        st.write(f"Boxplot de {variable}")
        fig, ax = plt.subplots()
        sns.boxplot(x=X[variable], ax=ax)
        st.pyplot(fig)

# Fonction pour l'analyse des corrélations

def viz_correlation(X):
    st.subheader("Analyse des Corrélations")
    corr_matrix = X.corr()

    st.write("Matrice de corrélation :")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    var1 = st.selectbox("Variable 1 :", options=list(X.columns))
    var2 = st.selectbox("Variable 2 :", options=list(X.columns))

    if var1 and var2:
        st.write(f"Scatter plot entre {var1} et {var2}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=X[var1], y=X[var2], ax=ax)
        st.pyplot(fig)

# Analyse de la target
def analyse_target(y):
    st.subheader("Analyse de la Target")
    st.write("Distribution de la target :")
    fig, ax = plt.subplots()
    sns.histplot(y, kde=True, ax=ax)
    st.pyplot(fig)

    if st.checkbox("Afficher le boxplot de la target"):
        fig, ax = plt.subplots()
        sns.boxplot(y, ax=ax)
        st.pyplot(fig)

# Graphique Interactifs avec Ploty
def interactive_plots(X, y):
    st.subheader("Graphiques Interactifs")

    # Histogramme interactif pour une variable sélectionnée
    variable = st.selectbox("Choisissez une variable pour le histogramme interactif :", options=X.columns)
    fig = px.histogram(X, x=variable, title=f"Histogramme Interactif de {variable}", marginal="box")
    st.plotly_chart(fig)

    # Scatter plot interactif pour deux variables
    var1 = st.selectbox("Variable X :", options=X.columns, key="plotly_var1")
    var2 = st.selectbox("Variable Y :", options=X.columns, key="plotly_var2")
    scatter_fig = px.scatter(X, x=var1, y=var2, title=f"Scatter Plot Interactif : {var1} vs {var2}", trendline="ols")
    st.plotly_chart(scatter_fig)


# Fonction principale pour appeler les fonctions d'analyse

def run_data_analysis(X, y):
    viz_analyse_descriptive(X)
    viz_correlation(X)
    analyse_target(y)
    interactive_plots(X, y)

#run_data_analysis(X, y)

"""def display_missing_data(X):
    st.subheader("Analyse des Données Manquantes")
    missing_data = X.isnull().sum().sort_values(ascending=False)
    st.write("Résumé des données manquantes :", missing_data)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(X.isnull(), cbar=False, cmap='viridis', ax=ax[0])
    ax[0].set_title("Heatmap des Valeurs Manquantes")

    missing_data.plot(kind='bar', ax=ax[1])
    ax[1].set_title("Nombre de Valeurs Manquantes par Colonne")

    st.pyplot(fig)
"""