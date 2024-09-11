import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
# Configuration de la page principale
#def config_page():
#    st.set_page_config(page_title="Préparation des Données", layout="wide")


# Fonction principale pour la gestion du prétraitement
def run_preprocessing_cls():
    #config_page()
    #colors = define_colors()
    df = load_dataset_option()

    if df is not None:
        #st.write("Informations sur le Dataset")
        display_data_overview(df)
        st.write(" Le nombre de lignes et de colonnes")
        st.write(pd.DataFrame({"Nombre de lignes": [df.shape[0]], "Nombre de colonnes": [df.shape[1]]}))
        st.write("Les types de données")
        st.write(pd.DataFrame(df.dtypes))

        selected_columns = select_columns(df)
        if selected_columns:
            clean_data(df, selected_columns)
            if st.checkbox("Encoder et télécharger le fichier traité"):
                download_processed_data(df)
        else:
            st.warning("Veuillez sélectionner des colonnes pour le traitement avant de continuer.")

# Fonction pour charger le fichier de vin ou un fichier propre
def load_dataset_option():
    option = st.radio("Choisissez un dataset:", ("Fichier vin", "Charger votre propre fichier CSV"))

    path = os.getcwd() + "/streamlit"

    if option == "Fichier vin":
        try:
            df = pd.read_csv(path + r"/data/vin.csv")
            # Assurez-vous que le fichier "vin.csv" est dans le bon répertoire
            st.success("Dataset vin chargé avec succès.")
            st.session_state['df'] = df # Initialiser st.session_state['df']
            return df
        except FileNotFoundError:
            st.error("Erreur : Le fichier 'vin.csv' est introuvable.")
            return None
    elif option == "Charger votre propre fichier CSV":
        uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
        if uploaded_file is not None:
            return load_data(uploaded_file)
    return None

# Fonction pour charger les données
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.write(f"Erreur lors du chargement du fichier : {e}")
        return None

# Fonction pour afficher l'aperçu des données et des informations
def display_data_overview(df):
    st.title("Préparation des Données")
    st.write("**Aperçu des données :**")
    st.write(df.head())
    st.write("**Informations sur le dataset :**")
    
    #st.write(f"Nombre de lignes : {df.shape[0]}")
    #st.write(f"Nombre de colonnes : {df.shape[1]}")

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
        #st.markdown(f'<div style="background-color:{colors["block_bg"]}; padding: 10px; border-radius: 5px;">',
                    #unsafe_allow_html=True)

        # Affichage des valeurs manquantes
        if st.checkbox("Afficher les valeurs manquantes"):
            st.write(df[selected_columns].isnull().sum())

        # Imputation des valeurs manquantes
        impute_missing_values(df, selected_columns)

        # Gestion des lignes/colonnes manquantes
        manage_missing_data(df, selected_columns)

        st.markdown('</div>', unsafe_allow_html=True)

# Fonction pour imputer les valeurs manquantes
def impute_missing_values(df, selected_columns):
    st.write("**Imputation des valeurs manquantes :**")
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
    st.write("**Gestion des valeurs manquantes :**")
    
    # Afficher les statistiques sur les valeurs manquantes
    missing_data = df[selected_columns].isnull().sum()
    st.write("**Valeurs manquantes par colonne :**")
    st.write(missing_data[missing_data > 0])

    # Choix de la méthode pour gérer les valeurs manquantes
    manage_method = st.radio(
        "Choisissez une méthode pour gérer les valeurs manquantes :",
        options=["Ne rien faire", "Supprimer les lignes", "Supprimer les colonnes", "Imputer des valeurs"]
    )

    if manage_method == "Supprimer les lignes":
        if st.button("Supprimer les lignes avec des valeurs manquantes"):
            df.dropna(subset=selected_columns, inplace=True)
            st.write("Lignes contenant des valeurs manquantes supprimées.")

    elif manage_method == "Supprimer les colonnes":
        if st.button("Supprimer les colonnes avec des valeurs manquantes"):
            df.dropna(axis=1, subset=selected_columns, inplace=True)
            st.write("Colonnes contenant des valeurs manquantes supprimées.")

    elif manage_method == "Imputer des valeurs":
        impute_strategy = st.selectbox(
            "Choisissez une méthode d'imputation des valeurs manquantes :",
            options=["Moyenne", "Médiane", "Valeur exacte"]
        )
        
        if impute_strategy == "Moyenne":
            if st.button("Imputer avec la moyenne"):
                imputer = SimpleImputer(strategy='mean')
                df[selected_columns] = imputer.fit_transform(df[selected_columns])
                st.write("Valeurs manquantes imputées avec la moyenne.")

        elif impute_strategy == "Médiane":
            if st.button("Imputer avec la médiane"):
                imputer = SimpleImputer(strategy='median')
                df[selected_columns] = imputer.fit_transform(df[selected_columns])
                st.write("Valeurs manquantes imputées avec la médiane.")

        elif impute_strategy == "Valeur exacte":
            exact_value = st.number_input("Entrez la valeur exacte pour imputer", value=0)
            if st.button("Imputer avec cette valeur"):
                imputer = SimpleImputer(strategy='constant', fill_value=exact_value)
                df[selected_columns] = imputer.fit_transform(df[selected_columns])
                st.write(f"Valeurs manquantes imputées avec la valeur exacte {exact_value}.")

    else:
        st.write("Aucune action n'a été choisie pour les valeurs manquantes.")



# Fonction pour modifier et encoder les données dans la colonne 'target'
def modify_and_encode_target(df):
    if 'target' in df.columns:
        # Correction des valeurs
        df['target'] = df['target'].replace('Vin éuilibré', 'Vin équilibré')
        
        # Affichage des valeurs distinctes avant encodage
        st.write("**Valeurs distinctes de la colonne 'target' après correction :**")
        st.write(df['target'].unique())
        
        # Encodage de la colonne 'target'
        le = LabelEncoder()
        df['target_encoded'] = le.fit_transform(df['target'])
        
        # Affichage des valeurs distinctes après encodage
        st.write("**Valeurs distinctes de la colonne 'target_encoded' :**")
        st.write(df['target_encoded'].unique())
        
        st.write("**Aperçu de la colonne 'target' après modification et encodage :**")
        #st.write(df[['target', 'target_encoded']].head())
    else:
        st.warning("La colonne 'target' n'est pas présente dans le DataFrame.")

# Fonction pour télécharger les données traitées
def download_processed_data(df):
    st.write("**Télécharger le fichier traité :**")
    
    # Modifier et encoder les données avant le téléchargement
    modify_and_encode_target(df)
    
    # Créer le CSV avec les modifications
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Télécharger le fichier CSV",
        data=csv,
        file_name='data_prepared_classification.csv',
        mime='text/csv'
    )
    
# Appel de la fonction principale pour le module de préprocessing
if __name__ == "__main__":
    run_preprocessing_cls()
