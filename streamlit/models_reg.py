
import streamlit as st
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Fonction principale pour la sous-page des modèles
def run_models(X, y):
    st.title("Sélection et Entraînement des Modèles de Régression")

    # Sélection des variables à inclure dans le modèle
    st.sidebar.header("Sélection de Features")
    selected_features = st.sidebar.multiselect("Choisissez les variables à inclure dans le modèle :", options=X.columns, default=list(X.columns))
    if selected_features:
        X = X[selected_features]  # Mise à jour des données avec les features sélectionnées
    else:
        X = st.session_state['X']

    y = st.session_state['y']

    # split des données
    split = X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Sélection du modèle
    st.subheader("Choix du Modèle de Régression")

    st.write("Taille de l'échantillon d'entrainement :", X_train.shape)
    st.write("Taille de l'échantillon test :", X_test.shape)

    model_choice = st.selectbox("Sélectionnez le modèle de régression :",
                                ["Régression Linéaire", "Lasso", "Random Forest", "Gradient Boosting"])


    # 2. Paramètres spécifiques au modèle sélectionné
    model = None
    params = {}

    if model_choice == "Régression Linéaire":
        model = LinearRegression()
        st.markdown("**Modèle sélectionné : Régression Linéaire**")

    elif model_choice == "Lasso":
        alpha = st.slider("Sélectionnez la valeur de l'alpha (régularisation) :", 0.01, 1.0, 0.1)
        model = Lasso(alpha=alpha)
        st.markdown(f"**Modèle sélectionné : Lasso Regression avec alpha = {alpha}**")
        params['alpha'] = alpha

    elif model_choice == "Random Forest":
        n_estimators = st.slider("Nombre d'arbres dans la forêt :", 10, 200, 100)
        max_depth = st.slider("Profondeur maximale des arbres :", 1, 20, 10)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        st.markdown(f"**Modèle sélectionné : Random Forest avec {n_estimators} arbres et profondeur max {max_depth}**")
        params['n_estimators'] = n_estimators
        params['max_depth'] = max_depth

    elif model_choice == "Gradient Boosting":
        learning_rate = st.slider("Taux d'apprentissage :", 0.01, 0.3, 0.1)
        n_estimators = st.slider("Nombre d'itérations de boosting :", 50, 200, 100)
        model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators)
        st.markdown(f"**Modèle sélectionné : Gradient Boosting avec taux d'apprentissage {learning_rate} et {n_estimators} itérations**")
        params['learning_rate'] = learning_rate
        params['n_estimators'] = n_estimators

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

    # Boutons de navigation
        # Boutons pour passer d'une page à l'autre
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Précédent : Analyse"):
            st.session_state.current_page = "Analyse"
            st.success(
                "Chargement de l'étape analyse de données !  \n Veuillez cliquer une deuxième fois pour l'afficher")  # Message d'information

    with col2:
        if st.button("Suivant : Évaluation"):
            st.session_state.current_page = "Évaluation"
            st.success(
                "Chargement de l'étape Evaluation !  \n Veuillez cliquer une deuxième fois pour l'afficher")  # Message d'information

# Fonction pour sauvegarder le modèle
def save_model(model, model_name, params):
    filename = f"{model_name}_model.pkl"
    joblib.dump(model, filename)
    st.success(f"Modèle {model_name} sauvegardé avec succès sous le nom {filename}.")
