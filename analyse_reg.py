import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.figure_factory as ff
import statsmodels.api as sm

# Fonction pour analyse descriptive

def viz_analyse_descriptive(X):
    st.subheader("Analyse Descriptive")
    variable = st.selectbox("Choisissez une variable à analyser :", options=list(X.columns))

    st.write(f"Histogramme de {variable}")
    fig, ax = plt.subplots()
    sns.histplot(X[variable], kde=True, ax=ax)
    st.pyplot(fig)

    if st.checkbox("Afficher le boxplot"):
        st.write(f"Boxplot de {variable}")
        fig, ax = plt.subplots()
        sns.boxplot(x=X[variable], ax=ax)
        st.pyplot(fig)

# Fonction pour l'analyse des corrélations

def viz_correlation(X):
    st.subheader("Analyse des Corrélations")
    corr_matrix = X.corr()

    st.write("Matrice de corrélation :")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    var1 = st.selectbox("Variable 1 :", options=list(X.columns))
    var2 = st.selectbox("Variable 2 :", options=list(X.columns))

    if var1 and var2:
        st.write(f"Scatter plot entre {var1} et {var2}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=X[var1], y=X[var2], ax=ax)
        st.pyplot(fig)

# Analyse de la target
def analyse_target(y):
    st.subheader("Analyse de la Target")
    st.write("Distribution de la target :")
    fig, ax = plt.subplots()
    sns.histplot(y, kde=True, ax=ax)
    st.pyplot(fig)

    if st.checkbox("Afficher le boxplot de la target"):
        fig, ax = plt.subplots()
        sns.boxplot(y, ax=ax)
        st.pyplot(fig)

# Graphique Interactifs avec Ploty
def interactive_plots(X, y):
    st.subheader("Graphiques Interactifs")

    # Histogramme interactif pour une variable sélectionnée
    variable = st.selectbox("Choisissez une variable pour le histogramme interactif :", options=X.columns)
    fig = px.histogram(X, x=variable, title=f"Histogramme Interactif de {variable}", marginal="box")
    st.plotly_chart(fig)

    # Scatter plot interactif pour deux variables
    var1 = st.selectbox("Variable X :", options=X.columns, key="plotly_var1")
    var2 = st.selectbox("Variable Y :", options=X.columns, key="plotly_var2")
    scatter_fig = px.scatter(X, x=var1, y=var2, title=f"Scatter Plot Interactif : {var1} vs {var2}", trendline="ols")
    st.plotly_chart(scatter_fig)


# Fonction principale pour appeler les fonctions d'analyse

def run_data_analysis(X, y):
    viz_analyse_descriptive(X)
    viz_correlation(X)
    analyse_target(y)
    interactive_plots(X, y)

#run_data_analysis(X, y)"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Fonction principale pour l'analyse des données
def run_data_analysis(X, y):
    st.title("Analyse des Données")

    # Expander pour l'overview des données avec histogrammes
    with st.expander("Overview des Données : Histogrammes"):
        fig, axes = plt.subplots(nrows=1, ncols=len(X.columns), figsize=(20, 5))
        for i, col in enumerate(X.columns):
            sns.histplot(X[col], ax=axes[i], kde=True)
            axes[i].set_title(f'Histogramme de {col}')
        st.pyplot(fig)

    # Expander pour les boxplots
    with st.expander("Distribution par Boxplot"):
        fig, axes = plt.subplots(nrows=1, ncols=len(X.columns), figsize=(20, 5))
        for i, col in enumerate(X.columns):
            sns.boxplot(x=X[col], ax=axes[i])
            axes[i].set_title(f'Boxplot de {col}')
        st.pyplot(fig)

    # Expander pour les Pairplots
    with st.expander("Pairplot des Variables"):
        sns.pairplot(X)
        st.pyplot()

    # Expander pour la matrice de corrélation
    with st.expander("Matrice de Corrélation"):
        corr_matrix = X.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Expander pour explorer les graphiques interactifs
    with st.expander("Exploration Interactive avec Plotly"):
        st.write("Voulez-vous une exploration interactive des données?")
        interactive = st.checkbox("Oui", value=False)
        if interactive:
            interactive_plots(X)

    # Expander pour l'analyse de la target
    with st.expander("Analyse de la Target"):
        analyze_target(y)

    # Expander pour vérifier la normalisation et proposer des options
    with st.expander("Vérification de la Normalisation des Données"):
        check_normalization(X)

    # Widgets de navigation entre les étapes
    st.markdown("---")
    st.write("Navigation entre les étapes :")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Étape Précédente : Prétraitement"):
            st.session_state.current_tab = "Prétraitement"
    with col2:
        if st.button("Étape Suivante : Modélisation"):
            st.session_state.current_tab = "Modélisation"


# Fonction pour les visualisations interactives
def interactive_plots(X):
    # Exemple d'un graphique interactif Plotly
    for col in X.columns:
        fig = px.histogram(X, x=col, title=f"Distribution de {col}", nbins=30)
        st.plotly_chart(fig, use_container_width=True)


# Fonction pour analyser la target
def analyze_target(y):
    st.write("**Distribution de la Variable Cible :**")
    fig = px.histogram(y, nbins=30, title="Histogramme de la Target")
    st.plotly_chart(fig)

    # Test de normalité
    stat, p_value = normaltest(y)
    st.write(f"Test de normalité (p-value = {p_value:.3f})")
    if p_value < 0.05:
        st.write("La distribution n'est pas normale. Cela peut affecter certains modèles de régression.")
        st.write("Modèles recommandés : Random Forest, Gradient Boosting, etc.")


# Fonction pour vérifier la normalisation et proposer la standardisation
def check_normalization(X):
    st.write("**Vérification de la Normalisation :**")
    non_normal_cols = [col for col in X.columns if normaltest(X[col])[1] < 0.05]

    if non_normal_cols:
        st.write("Certaines variables ne sont pas normalement distribuées :")
        st.write(non_normal_cols)
        st.write("Cela peut affecter les modèles de régression. Vous pouvez normaliser ou standardiser les données.")

        # Proposer des options de standardisation
        st.write("**Options de Standardisation :**")
        method = st.radio("Choisissez une méthode :", ["StandardScaler", "MinMaxScaler"])

        if st.button("Appliquer la standardisation"):
            if method == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            st.session_state['df'] = pd.concat([X_scaled, y], axis=1)
            st.write("Standardisation appliquée avec succès.")

    else:
        st.write("Toutes les variables sont normalement distribuées.")


# Appel de la fonction principale si le module est exécuté directement
if __name__ == "__main__":
    run_data_analysis(st.session_state['df'].drop(columns=['target']), st.session_state['df']['target'])

"""def display_missing_data(X):
    st.subheader("Analyse des Données Manquantes")
    missing_data = X.isnull().sum().sort_values(ascending=False)
    st.write("Résumé des données manquantes :", missing_data)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(X.isnull(), cbar=False, cmap='viridis', ax=ax[0])
    ax[0].set_title("Heatmap des Valeurs Manquantes")

    missing_data.plot(kind='bar', ax=ax[1])
    ax[1].set_title("Nombre de Valeurs Manquantes par Colonne")

    st.pyplot(fig)
"""