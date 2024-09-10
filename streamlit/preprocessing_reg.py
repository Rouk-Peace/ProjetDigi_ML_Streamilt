
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


def run_preprocessing():
    st.header("Prétraitement des Données")
    #st.write("Voici les étapes de prétraitement à réaliser...")
    # Ajoute ici les widgets spécifiques et la logique de prétraitement


    df = load_dataset_option()

    if df is not None:
        st.write("Informations sur le Dataset")
        display_data_overview(df)
        st.write(" Le nombre de lignes et de colonnes")
        st.write(pd.DataFrame({"Nombre de lignes": [df.shape[0]], "Nombre de colonnes": [df.shape[1]]}))
        st.write("Les types de données")
        st.write(pd.DataFrame(df.dtypes))

        selected_columns = select_columns(df)

        if selected_columns:
            clean_data(df, selected_columns)
            st.session_state['preprocessing_done'] = True  # Marque cette étape comme complétée
            #st.success("Prétraitement terminé avec succès.")

        else:
            st.warning("Veuillez sélectionner des colonnes pour le traitement avant de continuer.")


        # Si l'utilisateur souhaite télécharger le fichier traité
        if st.checkbox("Voulez-vous télécharger le fichier traité ?") and st.session_state.get('preprocessing_done',
                                                                                               False):
            download_processed_data(df)

        # Boutons de navigation

        navigation()

    else:
        st.warning("Veuillez charger un dataset pour démarrer le prétraitement.")

# Fonction pour charger le fichier de diabète ou un fichier propre
def load_dataset_option():
    path = os.getcwd() + "/streamlit"
   
    option = st.radio("Choisissez un dataset:", ("Fichier Diabète", "Charger votre propre fichier CSV"))
    if option == "Fichier Diabète":
        try:
            df = pd.read_csv(path + r"/data/diabete.csv")
  # Assurez-vous que le fichier "diabetes.csv" est dans le bon répertoire
            st.write("Dataset Diabète chargé avec succès.")
            st.session_state['df'] = df # Initialiser st.session_state['df']
            return df
        except FileNotFoundError:
            st.error("Erreur : Le fichier 'diabetes.csv' est introuvable.")
            return None
    elif option == "Charger votre propre fichier CSV":
        uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
        if uploaded_file is not None:
            return load_data(uploaded_file)
    return None


# Fonction pour charger les données
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Fichier chargé avec succès.")
        return df
    except Exception as e:
        st.write(f"Erreur lors du chargement du fichier : {e}")
        return None

# Fonction pour afficher l'aperçu des données et des informations
def display_data_overview(df):
    st.write("**Aperçu des données :**")
    st.write(df.head())

# Fonction pour sélectionner les colonnes pour le traitement
def select_columns(df):
    return st.sidebar.multiselect(
        "Sélectionnez les colonnes pour traitement",
        df.columns.tolist(),
        default=df.columns.tolist()
    )


# Fonction pour nettoyer les données : gestion des valeurs manquantes et encodage
def clean_data(df, selected_columns):
    with st.expander("Nettoyage des Données", expanded=True):

        # Affichage des valeurs manquantes
        st.write("Valeurs manquantes")
        st.write(df[selected_columns].isnull().sum())

        if df[selected_columns].isnull().sum().sum() == 0:
            st.write("Vous n'avez pas de valeurs manquantes.")
            if st.checkbox("Afficher les options de nettoyage"):
                show_data_cleaning_options(df, selected_columns)
        else:
            st.write("**Imputation des valeurs manquantes :**")
            st.write(df[selected_columns].isnull().sum())
            show_data_cleaning_options(df, selected_columns)
            st.markdown('</div>', unsafe_allow_html=True)

        #Sauvegarder le DataFrame nettoyé dans le session_state
        st.session_state['df_cleaned'] = df[selected_columns]  # Sauvegarde du DataFrame après nettoyage

# Fonction pour montrer les options de nettoyage des données
def show_data_cleaning_options(df, selected_columns):
    impute_missing_values(df, selected_columns)
    manage_missing_data(df, selected_columns)


# Fonction pour imputer les valeurs manquantes
def impute_missing_values(df, selected_columns):
    imputation_strategy = st.selectbox(
        "Choisissez la méthode d'imputation",
        ["Aucune", "Moyenne", "Médiane", "Valeur exacte"]
    )

    if imputation_strategy == "Moyenne" and st.button("Imputer avec la moyenne"):
        imputer = SimpleImputer(strategy='mean')
        df[selected_columns] = imputer.fit_transform(df[selected_columns])
        st.write("Valeurs manquantes imputées avec la moyenne.")

    elif imputation_strategy == "Médiane" and st.button("Imputer avec la médiane"):
        imputer = SimpleImputer(strategy='median')
        df[selected_columns] = imputer.fit_transform(df[selected_columns])
        st.write("Valeurs manquantes imputées avec la médiane.")

    elif imputation_strategy == "Valeur exacte":
        value = st.number_input("Entrez la valeur pour l'imputation", value=0)
        if st.button("Imputer avec la valeur exacte"):
            imputer = SimpleImputer(strategy='constant', fill_value=value)
            df[selected_columns] = imputer.fit_transform(df[selected_columns])
            st.write(f"Valeurs manquantes imputées avec la valeur {value}.")


# Fonction pour gérer les valeurs manquantes
def manage_missing_data(df, selected_columns):
    if st.button("Supprimer les lignes avec des valeurs manquantes"):
        df.dropna(subset=selected_columns, inplace=True)
        st.write("Lignes contenant des valeurs manquantes supprimées.")

    if st.button("Supprimer les colonnes avec des valeurs manquantes"):
        df.dropna(axis=1, subset=selected_columns, inplace=True)
        st.write("Colonnes contenant des valeurs manquantes supprimées.")


# Fonction pour télécharger les données traitées
def download_processed_data(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger le fichier CSV",
        data=csv,
        file_name='data_prepared.csv',
        mime='text/csv'
    )


# Fonction pour la navigation
def navigation():
    if st.button("Etape suivante: Analyse de données"):
        st.session_state.current_page = "Analyse"
        st.success("Chargement de l'étape analyse !  \n Veuillez cliquer une deuxième fois pour l'afficher")  # Message d'information


# Appel de la fonction principale pour le module de préprocessing
if __name__ == "__main__":
    run_preprocessing()
