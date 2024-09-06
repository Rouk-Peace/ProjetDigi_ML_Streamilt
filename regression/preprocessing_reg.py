import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

"""
# Configuration de la page principale

#def config_page():
    #st.set_page_config(page_title="Préparation des Données", layout="wide")



# Fonction principale pour la gestion du prétraitement
def run_preprocessing():
    #config_page()
    
    uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
                
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            display_data_overview(df)
            selected_columns = select_columns(df)

            if selected_columns:
                clean_data(df, selected_columns)
                download_processed_data(df)
                st.session_state['df'] = df  # Stocke les données dans la session
            else:
                st.write("Veuillez sélectionner des colonnes pour le traitement.")


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
    #buffer = st.empty()  # Pour afficher l'info du dataset proprement
    #buffer.text(df.info())  # Peut nécessiter un hack pour mieux afficher sur Streamlit

    st.write(f"**Nombre de lignes** : {df.shape[0]}")
    st.write(f"**Nombre de colonnes** : {df.shape[1]}")


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

        # Vérification des types de données
        if st.checkbox("Afficher les types de données actuels"):
            st.write(df.dtypes)

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
    if st.button("Supprimer les lignes avec des valeurs manquantes"):
        df.dropna(subset=selected_columns, inplace=True)
        st.write("Lignes contenant des valeurs manquantes supprimées.")

    if st.button("Supprimer les colonnes avec des valeurs manquantes"):
        df.dropna(axis=1, subset=selected_columns, inplace=True)
        st.write("Colonnes contenant des valeurs manquantes supprimées.")


# Fonction pour télécharger les données traitées
def download_processed_data(df):
    st.write("**Télécharger le fichier traité :**")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger le fichier CSV",
        data=csv,
        file_name='data_prepared.csv',
        mime='text/csv'
    )


# Appel de la fonction principale pour le module de préprocessing
#if __name__ == "__main__":
    #preprocessing_module()

"""




# Configuration de la page principale
#def config_page():
    #st.set_page_config(page_title="Préparation des Données", layout="wide")


# Définition des couleurs
def define_colors():
    return {
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


# Fonction principale pour la gestion du prétraitement
def run_preprocessing():
    #config_page()
    colors = define_colors()
    df = load_dataset_option()

    if df is not None:
        with st.expander("Informations sur le Dataset"):
            display_data_overview(df)
            if st.button("Afficher le nombre de lignes et de colonnes"):
                st.write(pd.DataFrame({"Nombre de lignes": [df.shape[0]], "Nombre de colonnes": [df.shape[1]]}))

            if st.button("Afficher les types de données"):
                st.write(df.dtypes)

        selected_columns = select_columns(df)

        if selected_columns:
            clean_data(df, selected_columns, colors)
            if st.checkbox("Voulez-vous télécharger le fichier traité ?"):
                download_processed_data(df)
        else:
            st.warning("Veuillez sélectionner des colonnes pour le traitement avant de continuer.")


# Fonction pour charger le fichier de diabète ou un fichier propre
def load_dataset_option():
    option = st.radio("Choisissez un dataset:", ("Fichier Diabète", "Charger votre propre fichier CSV"))
    if option == "Fichier Diabète":
        try:
            df = pd.read_csv(r"C:\Users\orouk\PycharmProjects\ProjetDigi_ML_Streamilt\data\diabete.csv")
  # Assurez-vous que le fichier "diabetes.csv" est dans le bon répertoire
            st.write("Dataset Diabète chargé avec succès.")
            st.session_state['df'] = df # Initialiser st.session_state['df']
            return df
        except FileNotFoundError:
            st.error("Erreur : Le fichier 'diabetes.csv' est introuvable.")
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
        st.write("Fichier chargé avec succès.")
        return df
    except Exception as e:
        st.write(f"Erreur lors du chargement du fichier : {e}")
        return None

# Fonction pour afficher l'aperçu des données et des informations
def display_data_overview(df):
    st.title("Préparation des Données")
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
def clean_data(df, selected_columns, colors):
    with st.expander("Nettoyage des Données", expanded=True):
        st.markdown(f'<div style="background-color:{colors["block_bg"]}; padding: 10px; border-radius: 5px;">',
                    unsafe_allow_html=True)

        # Affichage des valeurs manquantes
        if df[selected_columns].isnull().sum().sum() == 0:
            if st.checkbox("Vous n'avez pas de valeurs manquantes. Souhaitez-vous afficher les options de nettoyage ?"):
                show_data_cleaning_options(df, selected_columns)
        else:
            st.write("**Imputation des valeurs manquantes :**")
            st.write(df[selected_columns].isnull().sum())
            show_data_cleaning_options(df, selected_columns)

        st.markdown('</div>', unsafe_allow_html=True)


# Fonction pour montrer les options de nettoyage des données
def show_data_cleaning_options(df, selected_columns):
    impute_missing_values(df, selected_columns)
    manage_missing_data(df, selected_columns)
    if st.checkbox("Afficher les types de données actuels"):
        st.write(df.dtypes)


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


# Appel de la fonction principale pour le module de préprocessing
if __name__ == "__main__":
    run_preprocessing()
