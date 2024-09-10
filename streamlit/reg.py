import streamlit as st
from preprocessing_reg import run_preprocessing
from analyse_reg import run_analyse
from models_reg import run_models
from evaluation_reg import run_evaluation

def main_reg():
    # Configuration de la page
    #st.set_page_config(page_title="Projet ML: Modèles de Regression", layout="wide")
    st.title("Régression : De la Préparation à l'Évaluation des Modèles")
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
        """Prépare les données en sélectionnant la variable cible et divisant le DataFrame en X et y."""
        if 'df_cleaned' in st.session_state:
            df = st.session_state['df_cleaned']
            columns = df.columns.tolist()

            # Définir l'index par défaut à 0
            default_index = 0

            # Vérifier si "target" est une colonne et définir l'index par défaut
            if "target" in columns:
                default_index = columns.index("target")

            # Sélection de la target avec un index par défaut
            target = st.sidebar.selectbox(
                "Sélectionnez la target pour continuer :",
                options=columns,
                index=default_index  # Utiliser l'index de "target" si présent
            )

            # Division des données en X et y
            X = df.drop(columns=[target])
            y = df[target]

            # Stocker X, y et target dans st.session_state pour une utilisation globale
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['target'] = target
            st.session_state['df_cleaned'] = df

            # Retourner X et y pour les utiliser dans les étapes suivantes
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
        if st.session_state.current_page == "Analyse" or st.session_state.current_page =="Modélisation":

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



# Appel de la fonction principale pour le module de préprocessing
if __name__ == "__main__":
    main_reg()
