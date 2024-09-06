import streamlit as st
import pandas as pd
from PIL import Image
import io

# Configuration de la page principale
st.set_page_config(page_title="PLAYGROUND ML", layout="wide", page_icon="🤖")

# Chemins vers les images
logo_path = '/Users/sabaaziri/workspace/ ProjetDigi_ML_Streamilt /ProjetDigi_ML_Streamilt/app.py'
background_image_path = "/Users/sabaaziri/workspace/ ProjetDigi_ML_Streamilt /ProjetDigi_ML_Streamilt/img/computer-technology-business-website-header.jpg.jpg"
team_image_path = '/Users/sabaaziri/workspace/ ProjetDigi_ML_Streamilt /ProjetDigi_ML_Streamilt/img/team work.jpg'

# Chemins vers les bannières
banners = {
    #"Accueil": "",
    #"Équipe": "",
    #"Classification": "",
    #"Régression": "/Users/sabaaziri/Downloads/regression_banner.jpg",
    "Nail's detection (optionnel)": "/Users/sabaaziri/workspace/ ProjetDigi_ML_Streamilt /ProjetDigi_ML_Streamilt/img/roboflow.png",
    #"Conclusion": "/Users/sabaaziri/Downloads/conclusion_banner.jpg"
}

# Chargement des datasets prédéfinis
def load_wine_data():
    return pd.read_csv("/Users/sabaaziri/workspace/ ProjetDigi_ML_Streamilt /ProjetDigi_ML_Streamilt/data/vin.csv")

def load_diabetes_data():
    return pd.read_csv("/Users/sabaaziri/workspace/ ProjetDigi_ML_Streamilt /ProjetDigi_ML_Streamilt/data/diabete.csv")

# Initialisation de Roboflow
def init_roboflow():
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
    project = rf.workspace().project("your-project-name")
    return project.version(1).model

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
    Notre équipe est composée de passionnés de science des données, chacun ayant un rôle clé dans la réalisation de ce projet. Nous avons travaillé non seulement sur les aspects techniques du machine learning mais aussi sur l'optimisation de l'expérience utilisateur pour rendre cette application aussi intuitive et engageante que possible.

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
    
    st.write("""
    ### Contributions et Tests 
    
    Chaque membre de notre équipe a joué un rôle crucial dans le développement technique ainsi que dans l'optimisation de l'interface utilisateur et de l'expérience utilisateur :

    - **Roukyatou Oumourou** a dirigé les efforts en science des données pour créer des modèles de classification et de régression robustes. Elle a également contribué à l'optimisation des visualisations pour rendre les résultats plus accessibles et compréhensibles.
    - **Nacer Messaoui** a travaillé sur le développement des algorithmes de machine learning et a supervisé l'intégration des modèles dans l'application. Il a également assuré la performance des modèles en les testant avec divers jeux de données.
    - **Issam Harchi** a été responsable de l'évaluation des modèles et de l'amélioration de leur précision. Il a également contribué à l'optimisation de l'interface utilisateur pour une meilleure expérience.
    - **Saba Aziri** a supervisé la mise en œuvre de l'interface utilisateur et a intégré des éléments de design interactifs pour rendre l'application visuellement attrayante et facile à utiliser. Il a également ajouté des fonctionnalités amusantes pour améliorer l'interaction.


    """)



# Classification
elif page == "Classification":
    st.title("Classification")
    st.write("Explorez différentes techniques de classification.")
    
    if data is not None:
        st.write("Aperçu du dataset 'Vin' :")
        st.dataframe(data.head())
        st.write("""
        **Description du dataset vin selectionnez le csv vin pour la classification :**
        """)
        if st.button("Lancer une classification"):
            st.write("Entraînement du modèle de classification...")
            # Ajoutez ici le code pour entraîner un modèle de classification
            # Exemple: utiliser un modèle de classification depuis scikit-learn

# Régression
elif page == "Régression":
    st.title("Régression")
    st.write("Analysez et modélisez les données à l'aide de techniques de régression.")
    
    if data is not None:
        st.write("Aperçu des données :")
        st.dataframe(data.head())
        st.write("""
        **Description du dataset Diabete selectionnez le csv diabete pour la regression :**
        """)
        if st.button("Lancer une régression"):
            st.write("Entraînement du modèle de régression...")
            # Ajoutez ici le code pour entraîner un modèle de régression
            # Exemple: utiliser un modèle de régression depuis scikit-learn

# Nail's detection (optionnel)
elif page == "Nail's detection (optionnel)":
    st.title("Nail's detection")
    
    st.write("""
    ### Intégration avec Roboflow pour la Détection d'Ongles
    
    Dans le cadre de ce projet, nous avons intégré **Roboflow**, une plateforme puissante pour la création et le déploiement de modèles de vision par ordinateur. Roboflow simplifie le processus de formation de modèles de détection d'objets en fournissant des outils conviviaux pour annoter des images, entraîner des modèles et déployer des solutions de détection en temps réel.
    
    Pour notre tâche spécifique, nous avons utilisé Roboflow pour développer un modèle capable de détecter des ongles dans des images. Vous pouvez télécharger une image ci-dessous pour tester la détection d'objets et voir comment le modèle fonctionne.
    """)
    
    


# Conclusion
elif page == "Conclusion":
    st.title("Conclusion")
    
    st.write("""
    ### Méthodologie
    Notre projet a été géré en utilisant la méthodologie Agile, qui nous a permis de travailler de manière itérative et incrémentale. Nous avons organisé des sprints pour définir des objectifs clairs et avons régulièrement évalué nos progrès pour nous assurer que nous restions sur la bonne voie.

    ### Outils
    - **Gestion de projet :** Nous avons utilisé GitHub pour le suivi des tâches, la gestion des versions, et la collaboration entre les membres de l'équipe. Les issues et les pull requests nous ont aidés à organiser le travail et à réviser les contributions.
    - **Développement :** Nous avons utilisé des environnements de développement intégrés (IDE) comme PyCharm et VSCode pour écrire et tester notre code.
    - **Visualisation des données :** Pour la visualisation et l'analyse des données, nous avons utilisé des bibliothèques Python comme Matplotlib, Seaborn et Plotly.
    - **Machine Learning :** Nous avons utilisé des bibliothèques telles que Scikit-Learn et TensorFlow pour l'implémentation et l'évaluation de nos modèles de Machine Learning.
    - **Détection d'objets :** Nous avons intégré Roboflow pour la détection d'objets dans les images.

    ### Algorithmes Utilisés
    - **Classification :** Nous avons exploré des modèles de classification comme la régression logistique, les forêts aléatoires (Random Forest) et les machines à vecteurs de support (SVM).
    - **Régression :** Nous avons utilisé des algorithmes de régression tels que la régression linéaire et la régression Ridge pour prédire des valeurs continues.
    - **Détection d'objets :** Pour le projet de détection d'objets, nous avons intégré des modèles pré-entraînés de détection d'objets comme YOLO (You Only Look Once) via Roboflow.

    ### Perspectives
    - **Amélioration des Modèles :** Nous prévoyons d'expérimenter avec d'autres algorithmes de Machine Learning pour améliorer les performances des modèles.
    - **Sources de Données :** L'intégration de nouvelles sources de données nous permettra de rendre nos modèles plus robustes et polyvalents.
    - **Déploiement :** La prochaine étape est le déploiement de nos modèles en production, éventuellement via une API pour des prédictions en temps réel.
    """)

# Footer simplifié
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .css-12ttj6m {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

