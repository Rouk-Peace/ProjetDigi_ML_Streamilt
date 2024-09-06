import streamlit as st
import pandas as pd
from PIL import Image
from reg import main_reg  # Import de la fonction main depuis reg.py

# Configuration de la page principale
st.set_page_config(page_title="PLAYGROUND ML", layout="wide", page_icon="🤖")

# Chemins vers les images
logo_path = 'C:/Users/ELite/Workspace ML/Projet_streamlit_ML/ProjetDigi_ML_Streamilt/img/Logo_Diginamic.jpg'
background_image_path = "C:/Users/ELite/Workspace ML/Projet_streamlit_ML/ProjetDigi_ML_Streamilt/img/computer-technology-business-website-header.jpg"
team_image_path = 'C:/Users/ELite/Workspace ML/Projet_streamlit_ML/ProjetDigi_ML_Streamilt/img/team work.jpg'

# Chemins vers les bannières
banners = {
    "Nail's detection (optionnel)": "C:/Users/ELite/Workspace ML/Projet_streamlit_ML/ProjetDigi_ML_Streamilt/img/roboflow.png",
}

# Chargement des datasets prédéfinis
def load_wine_data():
    return pd.read_csv("C:/Users/ELite/Workspace ML/Projet_streamlit_ML/ProjetDigi_ML_Streamilt/data/vin.csv")

def load_diabetes_data():
    return pd.read_csv("C:/Users/ELite/Workspace ML/Projet_streamlit_ML/ProjetDigi_ML_Streamilt/data/diabete.csv")

# Sidebar: Logo et options
st.sidebar.image(logo_path, width=195)
st.sidebar.title("Sommaire")

# Choix du dataset
dataset_options = ["Vin", "Diabète", "Téléverser un fichier CSV"]
dataset_choice = st.sidebar.selectbox("Choisissez un dataset ou téléversez votre propre fichier :", dataset_options)

if dataset_choice == "Vin":
    data = load_wine_data()
    st.sidebar.success("Dataset 'Vin' chargé avec succès.")
elif dataset_choice == "Diabète":
    data = load_diabetes_data()
    st.sidebar.success("Dataset 'Diabète' chargé avec succès.")
else:
    uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Fichier CSV chargé avec succès !")
    else:
        data = None
        st.sidebar.info("Veuillez télécharger un fichier CSV pour commencer et tester les fonctionnalités.")

# Définition du sommaire dans la sidebar
pages = ["Accueil", "Équipe", "Classification", "Régression", "Nail's detection (optionnel)", "Conclusion"]
page = st.sidebar.radio("Naviguez vers :", pages)

# Affichage de la bannière pour chaque page
banner_path = banners.get(page)
if banner_path:
    st.image(banner_path, use_column_width=True)

# Accueil
if page == "Accueil":
    st.title("Playground Machine Learning 🤖")
    st.image(background_image_path, use_column_width=True)
    st.write("""
    Bienvenue dans l'application dédiée à l'analyse et à la modélisation de données.
    
    Vous pouvez Naviguez à l'aide du menu à gauche pour explorer toutes les sections :
    
    - **Classification** : Explorez différents modèles pour des problèmes de classification.
    - **Régression** : Analysez des modèles de régression pour les prédictions continues.
    - **Nail's detection** (optionnel) : Détection d'objets dans des images à l'aide de modèles de vision.
    """)

# Équipe
elif page == "Équipe":
    st.title("Présentation de l'équipe")
    
    st.write("""
    Notre équipe est composée de passionnés de science des données, chacun ayant un rôle clé dans la réalisation de ce projet.
    Voici les membres de notre équipe, leurs spécialités et leurs contributions :
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(team_image_path)
        st.image(image, caption="Équipe Data_Science", use_column_width=True)

    with col2:
        team_members = {
            "Roukyatou Oumourou : Data Scientist": "https://www.linkedin.com/in/roukayatou-omorou/",
            "Nacer Messaoui : Data Scientist": "https://www.linkedin.com/in/nacer-messaoui/",
            "Issam Harchi : Data Scientist": "https://www.linkedin.com/in/issam-harchi-a939b9100/",
            "Saba Aziri : Data Analyst": "https://www.linkedin.com/in/azirisaba/"
        }
        
        for name, link in team_members.items():
            st.markdown(f"- [{name}]({link})")

# Classification
elif page == "Classification":
    st.title("Classification")
    st.write("Explorez différentes techniques de classification.")
    
    if data is not None:
        st.write("Aperçu du dataset sélectionné :")
        st.dataframe(data.head())

    # Sous-sections pour Classification
    with st.expander("Prétraitement des Données"):
        st.write("Prétraitement des données pour la classification :")
        
    with st.expander("Analyse et Visualisation"):
        st.write("Visualisation des résultats de classification :")
        
    with st.expander("Modélisation"):
        st.write("Modélisation des données :")
        
    with st.expander("Évaluation"):
        st.write("Évaluation des modèles de classification :")

# Régression
elif page == "Régression":
    st.title("Régression")
    st.write("Explorez différentes techniques de régression.")

    if data is not None:
        st.write("Aperçu du dataset sélectionné :")
        st.dataframe(data.head())
        
        # Définir X et y pour la régression
        X = data.drop(columns=['target'])  # Assurez-vous que 'target' est bien la colonne de la variable cible
        y = data['target']
        
        # Appel de la fonction main() depuis le fichier reg.py
        main()

# Nail's detection (optionnel)
elif page == "Nail's detection (optionnel)":
    st.title("Nail's detection")
    st.write("""
    Intégration avec Roboflow pour la Détection d'Ongles
    """)
    
# Conclusion
elif page == "Conclusion":
    st.title("Conclusion")
    
    st.write("""
    ### Méthodologie
    Notre projet a été géré en utilisant la méthodologie Agile, avec des sprints réguliers.
    """)
    
# Footer simplifié
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .css-12ttj6m {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
