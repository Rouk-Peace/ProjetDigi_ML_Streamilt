import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
 
# Titre de la sous-page
st.title("Analyse de Données : Exploration et analyses visuelles des données")
 
# Phrase explicative d'en-tête
st.markdown("""
Bienvenue sur la page d'analyse des données. Cette section vous permet d'explorer, visualiser et comprendre
les caractéristiques du jeu de données avant la modélisation. L'objectif est de mettre en évidence les relations entre les variables,
d'identifier les tendances et les anomalies, et de préparer les données pour les étapes de modélisation.
""")
 
# Fonction pour l'analyse descriptive
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
 
# Fonction pour l'analyse des corrélations sur les variables numériques uniquement
def viz_correlation(X):
    st.subheader("Analyse des Corrélations")
 
    # Sélection des variables numériques uniquement
    X_numeric = X.select_dtypes(include=[np.number])
 
    if X_numeric.empty:
        st.write("Aucune variable numérique dans les données.")
    else:
        # Matrice de corrélation
        corr_matrix = X_numeric.corr()
 
        st.write("Matrice de corrélation (variables numériques uniquement) :")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
 
        var1 = st.selectbox("Variable 1 :", options=list(X_numeric.columns))
        var2 = st.selectbox("Variable 2 :", options=list(X_numeric.columns))
 
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
 
# Fonction pour l'évaluation des modèles de régression
def evaluation_modele_regression(y_true, y_pred):
    st.subheader("Évaluation du Modèle (Régression)")
    st.write("### Métriques de performance")
 
    # Calcul des métriques
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
 
    # Affichage des métriques
    st.write(f"Mean Absolute Error (MAE) : {mae}")
    st.write(f"Mean Squared Error (MSE) : {mse}")
    st.write(f"Root Mean Squared Error (RMSE) : {rmse}")
    st.write(f"R² Score : {r2}")
 
    # Affichage du scatter plot des valeurs prédites vs valeurs réelles
    st.write("### Valeurs réelles vs valeurs prédites")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    st.pyplot(fig)
 
# Fonction pour l'évaluation des modèles de classification
def evaluation_modele_classification(y_true, y_pred, class_names):
    st.subheader("Évaluation du Modèle (Classification)")
    st.write("### Rapport de Classification")
 
    # Afficher le rapport de classification sous forme de dataframe
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
 
# Fonction principale pour appeler les fonctions d'analyse
def run_data_analysis(X, y, y_pred=None, model_type='regression', class_names=None):
    viz_analyse_descriptive(X)
    viz_correlation(X)
    analyse_target(y)
 
    if model_type == 'regression' and y_pred is not None:
        evaluation_modele_regression(y, y_pred)
    elif model_type == 'classification' and y_pred is not None and class_names is not None:
        evaluation_modele_classification(y, y_pred, class_names)
 
# Exemple d'utilisation
if 'df' in st.session_state:
    df = st.session_state['df']  # Charger les données depuis session_state
    X = df.drop(columns=['target'])  # Variables explicatives
    y = df['target']  # Variable cible
 
    # Définir le type de modèle (régression ou classification)
    model_type = st.radio("Sélectionnez le type de modèle :", ('regression', 'classification'))
 
    # Si c'est un problème de régression
    if model_type == 'regression':
        y_pred = np.random.rand(len(y))  # Exemple de prédictions aléatoires pour la régression
        run_data_analysis(X, y, y_pred, model_type='regression')
 
    # Si c'est un problème de classification
    elif model_type == 'classification':
        y_pred = np.random.choice(np.unique(y), size=len(y))  # Exemple de prédictions aléatoires pour la classification
        class_names = ['Classe 0', 'Classe 1', 'Classe 2']  # Exemple de noms de classes
        run_data_analysis(X, y, y_pred, model_type='classification', class_names=class_names)
else:
    st.write("Les données ne sont pas chargées.")
