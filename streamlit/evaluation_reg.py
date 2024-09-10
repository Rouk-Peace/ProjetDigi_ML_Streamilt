import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Fonction principale pour la sous-page d'évaluation
def run_evaluation():
    st.title("Évaluation des Modèles")

    if 'model' not in st.session_state:
        st.warning("Aucun modèle entraîné trouvé. Veuillez entraîner un modèle dans la sous-page 'Modèles'.")
        return

    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']

    # 4. Évaluation du modèle
    st.subheader("Métriques d'Évaluation")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Affichage des métriques avec des explications contextuelles
    st.subheader("Les métriques d'évaluation")
    st.write(f"**RMSE :** {rmse:.4f} - Erreur quadratique moyenne entre les valeurs réelles et prédites.")
    st.write(f"**MAE :** {mae:.4f} - Erreur moyenne absolue, une mesure simple des erreurs moyennes.")
    st.write(f"**R² :** {r2:.4f} - Coefficient de détermination indiquant la proportion de variance expliquée.")

    # Graphique des valeurs réelles vs prédites
    st.subheader("Graphique des Valeurs Réelles vs Prédites")
    st.write("Ce graphique montre comment les prédictions du modèle se comparent aux valeurs réelles. Un bon modèle aura des points proches de la ligne rouge")
    plot_real_vs_predicted(y_test, y_pred)

    # Graphique de la distribution des résidus
    st.subheader("Distribution des Résidus")
    st.write("L’histogramme des résidus montre la distribution des erreurs. Idéalement, les résidus doivent suivre une distribution normale centrée autour de zéro. Cela indique que les erreurs sont aléatoires, sans biais systématique. Le modèle semble bien ajusté") 
    plot_residuals(y_test, y_pred)
    
    # Affichage des coefficients si modèle linéaire
    if isinstance(model, (LinearRegression, Lasso)):
        plot_coefficients(model, X_test.columns)

    # Boutons de navigation et comparaison des modèles
    if st.button("Comparer les modèles de regression"):
         # Récupération de X et y depuis st.session_state
        if 'X' in st.session_state and 'y' in st.session_state:
            X = st.session_state['X']
            y = st.session_state['y']
            compare_models(X, y)
        else:
            st.error("Les données X et y ne sont pas disponibles. Veuillez vérifier le prétraitement.")

    # Boutons de navigation
    if st.button("Etape précédente:Modélisation"):
        st.session_state.current_page = "Modélisation"
        st.success(
            "Chargement de l'étape Modélisation !  \n Veuillez cliquer une deuxième fois pour l'afficher")  # Message d'information


# Fonction pour afficher le graphique des valeurs réelles vs prédites
def plot_real_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Valeurs Réelles')
    ax.set_ylabel('Valeurs Prédites')
    ax.set_title('Valeurs Réelles vs Prédites')
    st.pyplot(fig)

#Fonction pour afficher le graphique des résidus
def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, ax=ax, color='purple', bins=30)
    ax.set_title("Distribution des Résidus")
    ax.set_xlabel("Résidus")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

# Fonction pour afficher les coefficients du modèle linéaire
def plot_coefficients(model, feature_names):
    st.subheader("Importance des Variables (Coefficients)")
    coef = pd.Series(model.coef_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=coef.values, y=coef.index, ax=ax, palette="coolwarm")
    plt.title("Coefficients des Variables")
    st.pyplot(fig)

def compare_models(X, y):
    """Compare différents modèles de régression et affiche les résultats.

    Args:
        X: Matrice des features
        y: Vecteur cible

    Returns:
        None
    """

    models = {
        'Régression Linéaire': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'Forêt Aléatoire': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'Lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [0.01, 0.1, 1, 10],
                'max_iter': [1000, 2000, 3000]
            }
        }
    }

    results = []
    best_models = {}

    for model_name, config in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config.get('params', {}),  # Gestion des cas où 'params' n'existe pas
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        best_models[model_name] = best_model

        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)


        results.append({
            'modèle': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Meilleurs paramètres': best_params
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Affichage du meilleur modèle
    best_model_name = results_df.loc[results_df['RMSE'].idxmin()]['modèle']
    st.write(f"Meilleur Modèle : {best_model_name} avec RMSE = {results_df['RMSE'].min():.4f}")
