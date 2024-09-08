import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Fonction principale pour la sous-page des modèles de classification
def run_model_selection(X, y):
    st.title("Sélection et Entraînement des Modèles de Classification")

    # Sélection des variables à inclure dans le modèle
    st.sidebar.header("Sélection de Features")
    selected_features = st.sidebar.multiselect("Choisissez les variables à inclure dans le modèle :", options=X.columns, default=list(X.columns))
    X = X[selected_features]  # Mise à jour des données avec les features sélectionnées

    # Sélection du modèle
    st.subheader("Choix du Modèle de Classification")
    
    model_choice = st.selectbox("Sélectionnez le modèle de classification :", 
                                ["Régression Logistique", "SVM", "Forêt Aléatoire"])

    model = None

    if model_choice == "Régression Logistique":
        model = LogisticRegression()
        st.markdown("**Modèle sélectionné : Régression Logistique**")

    elif model_choice == "SVM":
        model = SVC()
        st.markdown("**Modèle sélectionné : SVM**")

    elif model_choice == "Forêt Aléatoire":
        n_estimators = st.slider("Nombre d'arbres dans la forêt :", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators)
        st.markdown(f"**Modèle sélectionné : Forêt Aléatoire avec {n_estimators} arbres**")


    # 3. Entraînement du modèle
    if st.button("Entraîner le Modèle"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        st.session_state['model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = model.predict(X_test)
        st.success("Modèle entraîné avec succès !")

        # Sauvegarde du modèle
        if st.checkbox("Sauvegarder le Modèle Entraîné"):
            save_model(model, model_choice, params)

# Fonction pour sauvegarder le modèle
def save_model(model, model_name, params):
    filename = f"{model_name}_model.pkl"
    joblib.dump(model, filename)
    st.success(f"Modèle {model_name} sauvegardé avec succès sous le nom {filename}.")
