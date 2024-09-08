import streamlit as st
from preprocessing_reg import run_preprocessing
from analyse_reg import run_analyse
from models_reg import run_models
from evaluation_reg import run_evaluation

# Configuration de la page
st.set_page_config(page_title="Projet ML: Modèles de Regression", layout="wide")

# Crée un dictionnaire de pages pour lier les noms aux fonctions
pages = {
    "Prétraitement": run_preprocessing,
    "Analyse": run_analyse,
    "Modélisation": run_models,
    "Évaluation": run_evaluation
}

# Initialiser l'état de la page si ce n'est pas déjà fait
if "current_page" not in st.session_state:
    st.session_state.current_page = "Prétraitement"
    st.session_state.preprocessing_done = False

# Fonction pour changer de page
def navigate_to(page):
    st.session_state.current_page = page

# Fonction pour préparer les données pour l'analyse
def prepare_data():
    if 'df_cleaned' in st.session_state:
        df = st.session_state['df_cleaned']
        target = st.sidebar.selectbox("Sélectionnez la target pour continuer :", options=df.columns)
        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    else:
        st.warning("Veuillez effectuer le prétraitement avant l'analyse.")
        return None, None



# Créer une barre latérale pour la navigation séquentielle
st.sidebar.title("Navigation entre les Etapes ")

# Boutons séquentiels dans la sidebar pour suivre le processus ML
if st.sidebar.button("1. Prétraitement"):
    navigate_to("Prétraitement")

if st.sidebar.button("2. Analyse"):
    navigate_to("Analyse")

if st.sidebar.button("3. Modélisation"):
    navigate_to("Modélisation")

if st.sidebar.button("4. Évaluation"):
    navigate_to("Évaluation")

# Appelle la fonction correspondante à la page sélectionnée
if st.session_state.current_page in pages:
    if st.session_state.current_page == "Analyse" or st.session_state.current_page == "Modélisation":

        X, y = prepare_data()
        if X is not None and y is not None:
            pages[st.session_state.current_page](X, y)
    else:
        if st.session_state.current_page == "Prétraitement":
            pages["Prétraitement"]()
            st.session_state.preprocessing_done = True  # Marquer le prétraitement comme terminé après exécution
        else:
            pages[st.session_state.current_page]()
else:

    st.write("Bienvenue sur le projet de Machine Learning. Sélectionne une section pour commencer.")

