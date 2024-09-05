import streamlit as st
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction principale pour la sous-page des modèles
def run_model_selection(X, y):
    st.title("Sélection et Entraînement des Modèles de Régression")

    # Sélection des variables à inclure dans le modèle
    st.sidebar.header("Sélection de Features")
    selected_features = st.sidebar.multiselect("Choisissez les variables à inclure dans le modèle :", options=X.columns, default=list(X.columns))
    X = X[selected_features]  # Mise à jour des données avec les features sélectionnées

    # 1. Sélection du modèle
    st.subheader("Choix du Modèle de Régression")
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
        y_pred = model.predict(X_test)

        # 4. Évaluation du modèle
        st.subheader("Évaluation du Modèle")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**RMSE :** {rmse:.4f}")
        st.write(f"**MAE :** {mae:.4f}")
        st.write(f"**R² :** {r2:.4f}")

        # Graphique des valeurs réelles vs prédites
        st.subheader("Graphique des Valeurs Réelles vs Prédites")
        plot_real_vs_predicted(y_test, y_pred)

        # Affichage des coefficients si modèle linéaire
        if model_choice in ["Régression Linéaire", "Lasso"]:
            plot_coefficients(model, X.columns)

        # Sauvegarde du modèle
        if st.checkbox("Sauvegarder le Modèle Entraîné"):
            save_model(model, model_choice, params)

# Fonction pour afficher le graphique des valeurs réelles vs prédites
def plot_real_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Valeurs Réelles')
    ax.set_ylabel('Valeurs Prédites')
    ax.set_title('Valeurs Réelles vs Prédites')
    st.pyplot(fig)

# Fonction pour afficher les coefficients du modèle linéaire
def plot_coefficients(model, feature_names):
    st.subheader("Importance des Variables (Coefficients)")
    coef = pd.Series(model.coef_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=coef.values, y=coef.index, ax=ax, palette="coolwarm")
    plt.title("Coefficients des Variables")
    st.pyplot(fig)

# Fonction pour sauvegarder le modèle
def save_model(model, model_name, params):
    filename = f"{model_name}_model.pkl"
    joblib.dump(model, filename)
    st.success(f"Modèle {model_name} sauvegardé avec succès sous le nom {filename}.")

# Exemple d'appel de la fonction principale
# Appelée avec les données X et y depuis votre script app.py
# run_model_selection(X, y)
