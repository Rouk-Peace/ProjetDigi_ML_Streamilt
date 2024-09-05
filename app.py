import streamlit as st
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import r2_score



# Configuration de la page principale
st.set_page_config(page_title="PLAYGROUND ML", layout="wide")

# Définition des couleurs basées sur celles de Diginamic
colors = {
    'french_gray': '#C7D2DA',  # Gris clair
    'indigo_dye': '#1E3D59',  # Bleu foncé
    'prussian_blue': '#243A58',  # Bleu nuit
    'dark_goldenrod': '#A78F41',  # Doré foncé
    'background': '#FFFFFF',  # Blanc pour les blocs
    'button_bg': '#1E3D59',  # Bleu foncé bouton
    'button_text': '#FFFFFF',  # Texte bouton blanc
    'button_hover': '#172A40',  # Bleu très foncé au survol
    'expander_bg': '#E8F0FE',  # Fond léger pour l'expandeur
    'title_text': '#1E3D59',  # Bleu foncé pour les titres
    'subtitle_text': '#A78F41',  # Doré pour les sous-titres
    'border_color': '#E0E0E0',  # Gris clair pour les bordures
    }

st.sidebar.title("Sommaire")
 
pages = ["Accueil",
    "Contexte",
    "Présentation de l'équipe",
    "Classification",
    "Régression",
    "Nail's détection",
    "Conclusion et perspectives"]
 
page = st.sidebar.radio("Aller vers la page :", pages)
 
# Sidebar
#st.sidebar.title("Navigation")
#page = st.sidebar.selectbox("Choisissez une page", [])

# Chemin vers les données
data_dir = "/Users/sabaaziri/Downloads/streamlit/data"  # Mettre à jour le chemin en fonction de ton environnement

# Contenu de la page principale
if page == "Accueil":
    st.title("PLAYGROUND MACHINE LEARNING")
    st.write("""
    Bienvenue dans cette application dédiée à l'analyse et à la modélisation de jeux de données.
    Utilisez le menu à gauche pour naviguer entre les différentes sections de l'application.
    """)

elif page == "Contexte":
    st.title("Contexte")
    st.write("""
    Cette application vise à explorer et modéliser des jeux de données liés au vin et au diabète.
    Le but est d'appliquer des techniques de machine learning pour obtenir des insights précieux et des prédictions.
    """)

elif page == "Présentation de l'équipe":
    st.title("Présentation de l'équipe")
    st.write("""
    Voici les membres de notre équipe de projet PLAYGROUND MACHINE LEARNING :
    - Rouky : Description du rôle
    - Issam : Description du rôle
    - Nacer : Description du rôle
    - Saba : Description du rôle
    """)

elif page == "Jeux de données":
    st.title("Jeux de données")
    dataset_name = st.selectbox("Choisissez un dataset", ["Vin", "Diabète"])
    data_path = os.path.join(data_dir, f"{dataset_name.lower()}.csv")
    try:
        data1 = pd.read_csv("/Users/sabaaziri/workspace/ ProjetDigi_ML_Streamilt /ProjetDigi_ML_Streamilt/data/diabete.csv")
        
        st.write("Aperçu des données :")
        st.write(data.head())
        st.write("Distribution des données :")
        st.bar_chart(data)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")

elif page == "Préparation et exploration des données":
    st.title("Préparation et Exploration des Données")
    dataset_name = st.selectbox("Choisissez un dataset", ["Vin", "Diabète"])
    data_path = os.path.join(data_dir, f"{dataset_name.lower()}.csv")
    try:
        data = pd.read_csv(data_path)
        st.write("Données avant le prétraitement :")
        st.write(data.head())
        if st.button("Supprimer les valeurs manquantes"):
            data_cleaned = data.dropna()
            st.write("Données après suppression des valeurs manquantes :")
            st.write(data_cleaned.head())
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {e}")

elif page == "Analyse et Visualisation":
    st.title("Analyse et Visualisation")
    dataset_name = st.selectbox("Choisissez un dataset", ["Vin", "Diabète"])
    data_path = os.path.join(data_dir, f"{dataset_name.lower()}.csv")
    try:
        data = pd.read_csv(data_path)
        st.write("Matrice de corrélation :")
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de la visualisation des données : {e}")

elif page == "Modélisation":
    st.title("Modélisation - Machine Learning")
    dataset_name = st.selectbox("Choisissez un dataset", ["Vin", "Diabète"])
    data_path = os.path.join(data_dir, f"{dataset_name.lower()}.csv")
    try:
        data = pd.read_csv(data_path)
        # Assumer que la colonne cible s'appelle 'target'. Modifier en fonction du dataset réel.
        X = data.drop("target", axis=1)
        y = data["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Précision du modèle : {accuracy:.2f}")
    except Exception as e:
        st.error(f"Erreur lors de la modélisation : {e}")

elif page == "Conclusion et perspectives":
    st.title("Conclusion et Perspectives")
    st.write("""
    ### Conclusion
    Nous avons exploré et modélisé les datasets Vin et Diabète en utilisant diverses techniques de machine learning.
    Les résultats montrent que ...

    ### Perspectives
    - Amélioration des modèles avec d'autres techniques.
    - Exploration de nouvelles sources de données.
    - Déploiement d'une API pour prédictions en temps réel.
    """)


elif page == "Roboflow":
    st.title("Roboflow")
    st.write("""
    Cette section est dédiée à l'intégration avec Roboflow pour le traitement et l'augmentation des données d'image.
    """)

