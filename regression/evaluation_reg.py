import streamlit as st
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fonction principale pour la sous-page d'évaluation
def run_model_evaluation():
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
    st.write(f"**RMSE :** {rmse:.4f} - Erreur quadratique moyenne entre les valeurs réelles et prédites.")
    st.write(f"**MAE :** {mae:.4f} - Erreur moyenne absolue, une mesure simple des erreurs moyennes.")
    st.write(f"**R² :** {r2:.4f} - Coefficient de détermination indiquant la proportion de variance expliquée.")

    # Graphique des valeurs réelles vs prédites
    st.subheader("Graphique des Valeurs Réelles vs Prédites")
    plot_real_vs_predicted(y_test, y_pred)

    # Affichage des coefficients si modèle linéaire
    if isinstance(model, (LinearRegression, Lasso)):
        plot_coefficients(model, X_test.columns)

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
