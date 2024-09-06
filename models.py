
# Nail's detection (optionnel)
elif page == "Nail's detection (optionnel)":
    st.title("Nail's detection")
    
    st.write("""
    ### Intégration avec Roboflow pour la Détection d'Ongles
    
    Dans le cadre de ce projet, nous avons intégré **Roboflow**, une plateforme puissante pour la création et le déploiement de modèles de vision par ordinateur. Roboflow simplifie le processus de formation de modèles de détection d'objets en fournissant des outils conviviaux pour annoter des images, entraîner des modèles et déployer des solutions de détection en temps réel.
    
    Pour notre tâche spécifique, nous avons utilisé Roboflow pour développer un modèle capable de détecter des ongles dans des images. Vous pouvez télécharger une image ci-dessous pour tester la détection d'objets et voir comment le modèle fonctionne.
    """)
    
def nail_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans la détection d'ongle")

    # Entrer la clé API et le modèle ID
    api_key = st.text_input("Entrez votre clé API", type="password")
    model_id = st.text_input("Entrez l'ID de votre modèle")

    # Initialiser le client seulement si les informations sont fournies
    if api_key and model_id:
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )

        # Charger une image locale ou à partir d'une URL
        image_source = st.radio("Source de l'image", ["Image locale", "URL de l'image"])
        
        if image_source == "Image locale":
            uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                # Si l'image locale est choisie, on l'encode pour l'inférence
                result = CLIENT.infer(uploaded_file, model_id=model_id)
        else:
            image_url = st.text_input("Entrez l'URL de l'image")
            if image_url:
                # Si une URL d'image est fournie
                result = CLIENT.infer(image_url, model_id=model_id)

        # Afficher les résultats de l'inférence
        if 'result' in locals():
            st.write("Prédictions :")
            for pred in result['predictions']:
                st.write(f"Confiance: {pred['confidence']}, Classe: {pred['class']}")
    else:
        st.warning("Veuillez entrer votre clé API et le modèle ID pour continuer.")

# Exécuter la fonction
if __name__ == "__main__":
    nail_page()
