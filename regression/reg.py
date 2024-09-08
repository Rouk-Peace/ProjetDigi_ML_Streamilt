import streamlit as st
import preprocessing_reg as prep
import analyse_reg as analyse
import models_reg as model
import evaluation_reg as eval

# Configuration de la page principale
#st.set_page_config(page_title="Régression : Prétraitement, Analyse, Modélisation et Évaluation", layout="wide")


def main_reg():
    # Crée les onglets pour chaque étape
    st.title("Régression : De la Préparation à l'Évaluation des Modèles")
    tabs = ["Prétraitement", "Analyse", "Modélisation", "Évaluation"]
    current_tab = st.sidebar.radio("Étapes du Processus", tabs)

    # Vérification si les données sont disponibles pour l'analyse et la modélisation
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        # Vérifiez que df contient des données avant de procéder
        if not df.empty:
            target = st.sidebar.selectbox("Choisissez la variable cible :", options=df.columns)

    # Séparer les features et la target pour les prochaines étapes
            X = df.drop(columns=[target])
            y = df[target]

    # Gérer les onglets et progression séquentielle
    if current_tab == "Prétraitement":
        prep.run_preprocessing()  # Appel de la fonction de prétraitement
        if 'df' in st.session_state:
            st.session_state['preprocessed'] = True  # Marquer comme complété


    elif current_tab == "Analyse":
        if st.session_state.get('preprocessed', False):
            if 'df' in st.session_state:
                X = st.session_state['df'].drop(columns=[target])
                y = st.session_state['df'][target]
            analyse.run_data_analysis(X, y)  # Appel de la fonction d'analyse
            if st.button("Passer à la modélisation"):
                st.session_state['analysis_done'] = True
        else:
            st.warning("Veuillez compléter l'étape de prétraitement pour continuer.")

    elif current_tab == "Modélisation":
        if st.session_state.get('analysis_done', False):
            model.run_model_selection(X, y)  # Appel de la fonction de modélisation
            if st.button("Passer à l'évaluation"):
                st.session_state['modeling_done'] = True
        else:
            st.warning("Veuillez compléter l'étape d'analyse pour continuer.")

    elif current_tab == "Évaluation":
        if st.session_state.get('modeling_done', False):
            eval.run_model_evaluation()  # Appel de la fonction d'évaluation
        else:
            st.warning("Veuillez compléter l'étape de modélisation pour continuer.")


if __name__ == "__main__":
    main_reg()
