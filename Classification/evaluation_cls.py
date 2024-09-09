import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
 
# Fonction principale pour la sous-page d'évaluation
def run_model_evaluation():
    st.title("Évaluation des Modèles")
 
    # Vérification que le modèle et les données de test existent
    if 'model' not in st.session_state:
        st.warning("Aucun modèle entraîné trouvé. Veuillez entraîner un modèle dans la sous-page 'Modèles'.")
        return
 
    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    y_pred = st.session_state.get('y_pred', None)
 
    # Vérifier que les prédictions ont été faites
    if y_pred is None:
        st.error("Aucune prédiction trouvée. Veuillez générer des prédictions avant de procéder à l'évaluation.")
        return
 
    # Appel de la fonction pour afficher le rapport de classification et la matrice de confusion
    show_classification_report(y_test, y_pred, model)
 
# Fonction pour afficher le rapport de classification et la matrice de confusion
def show_classification_report(y_test, y_pred, model):
    # Rapport de classification
    st.subheader(f"Rapport de Classification pour {model.__class__.__name__}")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)  # Utilisation de dataframe pour un meilleur affichage
 
    # Matrice de confusion
    st.subheader(f"Matrice de Confusion pour {model.__class__.__name__}")
    cm = confusion_matrix(y_test, y_pred)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Matrice de Confusion - {model.__class__.__name__}")
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Valeurs Réelles")
    st.pyplot(fig)
 
# Appel de la fonction principale pour l'évaluation
run_model_evaluation()