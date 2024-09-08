import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Affiche le titre et l'en-tête explicatif de la page
def display_title_and_header():
    st.header("Analyse des Données")
    st.markdown("""
    Bienvenue dans l'étape d'analyse des données. Cette section vous permet d'explorer, visualiser et comprendre
    les caractéristiques du jeu de données avant la modélisation. 
    """)

# Sélectionne la variable cible et divise les données en X et y
def select_target_variable(df):
    """Sélectionne la variable cible et divise les données en X et y."""
    target = st.selectbox("Sélectionnez la target :", options=df.columns, key="target_select")
    st.session_state["target"] = target  # Stocker le nom de la variable cible
    X = df.drop(columns=[target])
    y = df[target]
    st.session_state['X'] = X  # Sauvegarder X dans le session_state
    st.session_state['y'] = y  # Sauvegarder y dans le session_state
    st.session_state['X_cols'] = df.columns.tolist()  # Stocker les noms originaux des colonnes
    return X, y


# Affiche des graphiques interactifs avec Plotly pour explorer les variables
def interactive_plots(X):
    """Affiche des graphiques interactifs avec Plotly pour explorer les variables."""
    with st.expander("Exploration Interactive avec Plotly"):
        st.write("Sélectionnez les variables à explorer de manière interactive.")

        # Affichage de l'histogramme avant le scatter plot
        hist_var = st.selectbox("Choisissez une variable pour l'histogramme interactif :", options=X.columns, key="hist_var")
        hist_fig = px.histogram(X, x=hist_var, title=f"Histogramme Interactif de {hist_var}", marginal="box")
        st.plotly_chart(hist_fig)

        # Vérification des colonnes disponibles pour les scatter plots
        #st.write("Variables disponibles pour les scatter plots :", X.columns.tolist())


        # Affichage du scatter plot après l'histogramme
        var1 = st.selectbox("Variable X :", options= X.columns, key="plotly_var1")
        var2 = st.selectbox("Variable Y :", options=X.columns, key="plotly_var2")

        # Vérification que var1 et var2 existent dans X
        if var1 in X.columns and var2 in X.columns:
            scatter_fig = px.scatter(X, x=var1, y=var2, title=f"Scatter Plot : {var1} vs {var2}", trendline="ols")
            st.plotly_chart(scatter_fig)
        else:
            st.warning(f"Les variables sélectionnées ({var1}, {var2}) ne sont pas valides.")


# Affiche les histogrammes et boxplots des variables de X
def overview_plots(X):
    """Affiche les histogrammes et boxplots des variables de X."""
    with st.expander("Overview des Données : Histogrammes et Boxplots"):
        st.write("Visualisation globale des distributions de toutes les variables.")
        fig, axes = plt.subplots(1, len(X.columns), figsize=(20, 5))
        for i, col in enumerate(X.columns):
            sns.histplot(X[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Histogramme de {col}')
        st.pyplot(fig)

        fig, axes = plt.subplots(1, len(X.columns), figsize=(20, 5))
        for i, col in enumerate(X.columns):
            sns.boxplot(x=X[col], ax=axes[i])
            axes[i].set_title(f'Boxplot de {col}')
        st.pyplot(fig)

# Affiche la matrice de corrélation des variables de X
def display_correlation_matrix(X):
    """Affiche la matrice de corrélation des variables de X."""
    with st.expander("Matrice de Corrélation"):
        corr_matrix = X.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Analyse la variable cible et affiche un histogramme avec un test de normalité
def analyze_target(y):
    """Analyse la variable cible et affiche un histogramme avec un test de normalité."""
    with st.expander("Analyse de la Target"):
        st.write("**Distribution de la Variable Cible :**")
        fig = px.histogram(y, nbins=30, title="Histogramme de la Target")
        st.plotly_chart(fig)

        # Test de normalité
        stat, p_value = normaltest(y)
        st.write(f"Test de normalité (p-value = {p_value:.3f})")
        if p_value < 0.05:
            st.write("La distribution n'est pas normale. Cela peut affecter certains modèles de régression.")
            st.write("Modèles recommandés : Random Forest, Gradient Boosting, etc.")

# Vérifie la normalité des données et propose des options de normalisation
def check_normalization(X):
    """Vérifie la normalité des données et propose des options de normalisation."""
    with st.expander("Vérification de la Normalisation des Données"):
        non_normal_cols = [col for col in X.columns if normaltest(X[col])[1] < 0.05]
        if non_normal_cols:
            st.write("Certaines variables ne sont pas normalement distribuées :")
            st.write(non_normal_cols)
            st.write("Cela peut affecter les modèles de régression. Vous pouvez normaliser ou standardiser les données.")
            method = st.radio("Choisissez une méthode :", ["StandardScaler", "MinMaxScaler"])
            if st.button("Appliquer la standardisation"):
                scaler = StandardScaler() if method == "StandardScaler" else MinMaxScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                st.session_state['df'] = pd.concat([X_scaled, y], axis=1)
                st.write("Standardisation appliquée avec succès.")
        else:
            st.write("Toutes les variables sont normalement distribuées.")

# Affiche les boutons de navigation pour passer d'une étape à l'autre
def navigation():
    """Affiche les boutons de navigation pour passer d'une étape à l'autre."""
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Étape précédente : Prétraitement"):
            st.session_state.current_page = "Prétraitement"
            st.success(
                "Chargement de l'étape Prétraitement !  \n Veuillez cliquer une deuxième fois pour l'afficher")  # Message d'information

    with col2:
        if st.button("Étape suivant : Modélisation"):
            st.session_state.current_page = "Modélisation"
            st.success(
                "Chargement de l'étape de modélisation !  \n Veuillez cliquer une deuxième fois pour l'afficher")  # Message d'information


# Point d'entrée principal pour exécuter les fonctions de l'analyse des données
def run_analyse(X, y):
    """Exécute les fonctions de l'analyse des données."""
    display_title_and_header()

    if 'df' in st.session_state and not st.session_state['df_cleaned'].empty:
        df = st.session_state['df_cleaned']
        #X, y = select_target_variable(df)
        st.session_state['X'] = X  # Sauvegarder X dans le session_state
        st.session_state['y'] = y
        interactive_plots(X)
        overview_plots(X)
        display_correlation_matrix(X)
        analyze_target(y)
        check_normalization(X)
        st.session_state['analyse_done'] = True
        navigation()
    else:
        st.warning("Veuillez charger les données avant de procéder à l'analyse.")
