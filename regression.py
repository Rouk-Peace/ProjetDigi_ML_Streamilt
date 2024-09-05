import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration de la page principale
st.set_page_config(page_title="PLAYGROUND ML", layout="wide")

# Titre de l'application
st.title("Machine Learning Playground")

# Sidebar - Interactions avec l'utilisateur
st.sidebar.header("Options")

# Upload du fichier CSV par l'utilisateur
uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Affichage des premières lignes du dataset
        st.write("**Aperçu des données :**")
        st.write(df.head())

        # Affichage des informations du dataset
        st.write("**Informations sur le dataset :**")
        buffer = pd.DataFrame({'Nombre de lignes': [df.shape[0]], 'Nombre de colonnes': [df.shape[1]]})
        st.table(buffer)
        st.write(df.info())

        # Sélection des colonnes pour X et y
        st.sidebar.subheader("Sélection des variables")
        target_col = st.sidebar.selectbox("Sélectionnez la colonne cible (y)", df.columns)
        feature_cols = st.sidebar.multiselect("Sélectionnez les colonnes de caractéristiques (X)", df.columns.drop(target_col))

        if target_col and feature_cols:
            X = df[feature_cols]
            y = df[target_col]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Visualisation des distributions et corrélations
            st.subheader("Visualisations")
            if st.checkbox("Afficher les boxplots"):
                for column in df.columns:
                    st.write(f"**Boxplot de la variable {column}**")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=df, x=column, ax=ax)
                    st.pyplot(fig)

            if st.checkbox("Afficher les histogrammes"):
                for column in df.columns:
                    st.write(f"**Histogramme de la variable {column}**")
                    fig, ax = plt.subplots()
                    sns.histplot(data=df, x=column, kde=True, ax=ax)
                    st.pyplot(fig)

            # Modèles sélectionnables
            st.sidebar.subheader("Sélection des modèles")
            models_to_run = st.sidebar.multiselect(
                "Choisissez les modèles à tester",
                ["Régression Linéaire", "Random Forest", "Gradient Boosting", "Lasso"],
                default=["Régression Linéaire"]
            )

            # Grilles de paramètres pour chaque modèle
            model_grids = {
                'Régression Linéaire': {
                    'model': LinearRegression(),
                    'params': {
                        'fit_intercept': [True, False]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 10]
                    }
                },
                'Lasso': {
                    'model': Lasso(),
                    'params': {
                        'alpha': [0.01, 0.1, 1.0, 10]
                    }
                }
            }

            # Fonction pour l'entraînement et l'évaluation des modèles
            def evaluate_model(model, X_train, X_test, y_train, y_test):
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Affichage des métriques d'évaluation
                st.write(f"### {type(model).__name__}")
                st.write("#### Performances sur le jeu de données d'entraînement")
                st.write(f"- MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
                st.write(f"- MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
                st.write(f"- R²: {r2_score(y_train, y_pred_train):.4f}")

                st.write("#### Performances sur le jeu de données de test")
                st.write(f"- MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
                st.write(f"- MSE: {mean_squared_error(y_test, y_pred_test):.4f}")
                st.write(f"- R²: {r2_score(y_test, y_pred_test):.4f}")

            # Parcourir les modèles sélectionnés par l'utilisateur
            for model_name in models_to_run:
                if model_name in model_grids:
                    model_info = model_grids[model_name]
                    model = model_info['model']
                    params = model_info['params']

                    # Application de GridSearchCV si l'utilisateur le souhaite
                    if st.sidebar.checkbox(f"Optimiser {model_name} avec GridSearchCV"):
                        st.write(f"Recherche des meilleurs hyperparamètres pour {model_name}...")
                        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        st.write(f"**Meilleurs hyperparamètres pour {model_name}:**")
                        st.write(grid_search.best_params_)
                        evaluate_model(best_model, X_train, X_test, y_train, y_test)
                    else:
                        # Entraînement du modèle sans optimisation
                        evaluate_model(model, X_train, X_test, y_train, y_test)

        else:
            st.warning("Veuillez sélectionner au moins une colonne cible (y) et des colonnes de caractéristiques (X).")

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")

else:
    st.write("Veuillez télécharger un fichier CSV pour commencer.")

