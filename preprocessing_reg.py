import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Configuration de la page principale
st.set_page_config(page_title="Préparation des Données", layout="wide")

# Définition des couleurs
colors = {
    'background': '#F5F5F5',
    'block_bg': '#FFFFFF',
    'text': '#1E3D59',
    'button_bg': '#1E3D59',
    'button_text': '#FFFFFF',
    'button_hover': '#172A40',
    'expander_bg': '#E8F0FE',
    'title_text': '#1E3D59',
    'subtitle_text': '#A78F41',
    'border_color': '#E0E0E0',
}

# Upload du fichier CSV par l'utilisateur
st.sidebar.header("Options de Prétraitement")
uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Affichage des premières lignes du dataset
        st.title("Préparation des Données")
        st.write("**Aperçu des données :**")
        st.write(df.head())

        # Affichage des informations du dataset
        st.write("**Informations sur le dataset :**")
        st.write(df.info())
        st.write(f"Nombre de lignes : {df.shape[0]}")
        st.write(f"Nombre de colonnes : {df.shape[1]}")

        # Sélection des colonnes
        selected_columns = st.sidebar.multiselect(
            "Sélectionnez les colonnes pour traitement",
            df.columns.tolist(),
            default=df.columns.tolist()
        )

        if selected_columns:
            # Nettoyage des Données
            with st.expander("Nettoyage des Données", expanded=True):
                st.markdown(f'<div style="background-color:{colors["block_bg"]}; padding: 10px; border-radius: 5px;">', unsafe_allow_html=True)

                # Affichage des valeurs manquantes
                if st.checkbox("Afficher les valeurs manquantes"):
                    st.write(df[selected_columns].isnull().sum())

                # Imputation des valeurs manquantes
                st.write("**Imputation des valeurs manquantes :**")
                imputation_strategy = st.selectbox(
                    "Choisissez la méthode d'imputation",
                    ["Aucune", "Moyenne", "Médiane", "Valeur exacte"]
                )

                if imputation_strategy == "Moyenne":
                    if st.button("Imputer avec la moyenne"):
                        imputer = SimpleImputer(strategy='mean')
                        df[selected_columns] = imputer.fit_transform(df[selected_columns])
                        st.write("Valeurs manquantes imputées avec la moyenne.")
                elif imputation_strategy == "Médiane":
                    if st.button("Imputer avec la médiane"):
                        imputer = SimpleImputer(strategy='median')
                        df[selected_columns] = imputer.fit_transform(df[selected_columns])
                        st.write("Valeurs manquantes imputées avec la médiane.")
                elif imputation_strategy == "Valeur exacte":
                    value = st.number_input("Entrez la valeur pour l'imputation", value=0)
                    if st.button("Imputer avec la valeur exacte"):
                        imputer = SimpleImputer(strategy='constant', fill_value=value)
                        df[selected_columns] = imputer.fit_transform(df[selected_columns])
                        st.write(f"Valeurs manquantes imputées avec la valeur {value}.")

                # Gestion des valeurs manquantes
                st.write("**Gestion des valeurs manquantes :**")
                if st.button("Supprimer les lignes avec des valeurs manquantes"):
                    df = df.dropna(subset=selected_columns)
                    st.write("Lignes contenant des valeurs manquantes supprimées.")
                if st.button("Supprimer les colonnes avec des valeurs manquantes"):
                    df = df.dropna(axis=1, subset=selected_columns)
                    st.write("Colonnes contenant des valeurs manquantes supprimées.")

                # Encodage des variables catégorielles
                st.write("**Encodage des variables catégorielles :**")
                if st.checkbox("Afficher les colonnes catégorielles"):
                    categorical_columns = df[selected_columns].select_dtypes(include=['object', 'category']).columns
                    st.write("Colonnes catégorielles détectées :", categorical_columns)

                encoding_option = st.selectbox(
                    "Choisissez la méthode d'encodage des variables catégorielles",
                    ["Aucune", "Encodage One-Hot"]
                )

                if encoding_option == "Encodage One-Hot":
                    if st.button("Encoder les variables catégorielles"):
                        df = pd.get_dummies(df, columns=categorical_columns)
                        st.write("Variables catégorielles encodées avec One-Hot.")

                # Vérification des types de données
                st.write("**Vérification des types de données :**")
                if st.checkbox("Afficher les types de données actuels"):
                    st.write(df.dtypes)

                st.markdown('</div>', unsafe_allow_html=True)

            # Téléchargement du fichier traité
            st.write("**Télécharger le fichier traité :**")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger le fichier CSV",
                data=csv,
                file_name='data_prepared.csv',
                mime='text/csv'
            )
        else:
            st.write("Veuillez sélectionner des colonnes pour le traitement.")
    except Exception as e:
        st.write(f"Erreur lors du chargement du fichier : {e}")

