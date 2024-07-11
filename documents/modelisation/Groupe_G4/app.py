import streamlit as st
import pandas as pd
import os
from Modèle_SAE import train_and_evaluate_model  # Assurez-vous que Modèle_SAE.py est dans le même répertoire
from Modèle_SAE import train_and_evaluate_model_tree
# Configuration de la mise en page de l'application
st.set_page_config(layout="wide")

# CSS pour centrer les en-têtes et les diagnostics
st.markdown("""
<style>
/* Centrer les en-têtes */
h1, h2, h3, h4, h5, h6 {
    text-align: center;
}

/* Ajuster la largeur des conteneurs pour centrer le formulaire et les résultats */
.container {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    height: 80vh;
    padding: 20px;
}

.left-column, .right-column {
    width: 45%;
    padding: 10px;
}

/* Centrer le contenu des diagnostics et des st.write */
.center-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.center-text {
    text-align: center;
}

/* Centrer les lignes des informations saisies */
.center-info {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-left: 20px;
    margin-top: 20px;
}

/* Centrer les diagnostics */
.center-diagnostic {
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    margin-top: 10px;
}

/* Centrer le graphique et le texte sous les résultats */
.center-plot {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Chargement des données avec mise en cache
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# Chargement initial du fichier CSV
filename = os.path.join("data", "data_origine_export.csv")
data = load_data(filename)

# Obtenir les valeurs uniques pour chaque caractéristique
modele_options = data['MODELE'].unique()
constructeur_options = data['CONSTRUCTEUR'].unique()
moteur_options = data['MOTEUR'].unique()
sig_organe_options = data['SIG_ORGANE'].unique()
sig_obs_options = data['SIG_OBS'].unique()
ligne_options = data['LIGNE'].unique()
kilometrage_classe_options = data['KILOMETRAGE_CLASSE'].unique()

# Affichage du titre
st.title("Interface de Diagnostic")

# Utilisation de colonnes pour disposer les éléments
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="left-column">', unsafe_allow_html=True)
    st.header("Entrer les informations du véhicule")
    with st.form(key='vehicle_form'):
        modele = st.selectbox("MODELE", modele_options)
        constructeur = st.selectbox("CONSTRUCTEUR", constructeur_options)
        moteur = st.selectbox("MOTEUR", moteur_options)
        sig_organe = st.selectbox("SIG_ORGANE", sig_organe_options)
        sig_obs = st.selectbox("SIG_OBS", sig_obs_options)
        ligne = st.selectbox("LIGNE", ligne_options)
        kilometrage_classe = st.selectbox("KILOMETRAGE_CLASSE", kilometrage_classe_options)
        
        # Bouton de soumission
        submit_button = st.form_submit_button(label='Diagnostiquer')

    if submit_button:
        # Filtrer les données selon les sélections
        filtered_data = data[(data['MODELE'] == modele) & 
                             (data['CONSTRUCTEUR'] == constructeur) & 
                             (data['MOTEUR'] == moteur) & 
                             (data['SIG_ORGANE'] == sig_organe) & 
                             (data['SIG_OBS'] == sig_obs) & 
                             (data['LIGNE'] == ligne) & 
                             (data['KILOMETRAGE_CLASSE'] == kilometrage_classe)]

        # Vérifier si des données sont disponibles après le filtrage
        if filtered_data.empty:
            st.error("Aucune donnée trouvée pour les sélections données. Veuillez ajuster les filtres.")
        else:
            # Afficher les détails du véhicule sélectionné
            st.header("Détails du véhicule sélectionné")
            st.write(filtered_data)

            # Vous pouvez ajouter d'autres fonctionnalités ou affichages ici si nécessaire

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="right-column">', unsafe_allow_html=True)
    st.header("Résultats des Modèles")

    # Exécution des modèles pour SYSTEM_N1, SYSTEM_N2, et SYSTEM_N3
    accuracies_nb = []
    accuracies_tree = []
    accuracies_mlp = []

    for target in ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"]:
        # Naive Bayes
        result_nb = train_and_evaluate_model(data, target, model_type='nb')
        if isinstance(result_nb, tuple):
            accuracy_nb = result_nb[0]
        else:
            accuracy_nb = result_nb
        accuracies_nb.append(accuracy_nb)

        # Decision Tree
        result_tree = train_and_evaluate_model(data, target, model_type='tree')
        if isinstance(result_tree, tuple):
            accuracy_tree = result_tree[0]
        else:
            accuracy_tree = result_tree
        accuracies_tree.append(accuracy_tree)

        # MLP
        result_mlp = train_and_evaluate_model(data, target, model_type='mlp')
        if isinstance(result_mlp, tuple):
            accuracy_mlp = result_mlp[0]
        else:
            accuracy_mlp = result_mlp
        accuracies_mlp.append(accuracy_mlp)

    st.subheader("Accuracies des Modèles")
    st.write("Naive Bayes:")
    for target, accuracy in zip(["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"], accuracies_nb):
        st.write(f"{target}: {accuracy}")

    st.write("Decision Tree:")
    for target, accuracy in zip(["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"], accuracies_tree):
        st.write(f"{target}: {accuracy}")

    st.write("MLP:")
    for target, accuracy in zip(["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"], accuracies_mlp):
        st.write(f"{target}: {accuracy}")
    
    filtered_data = data[(data['MODELE'] == modele) & 
                         (data['CONSTRUCTEUR'] == constructeur) & 
                         (data['MOTEUR'] == moteur) & 
                         (data['SIG_ORGANE'] == sig_organe) & 
                         (data['SIG_OBS'] == sig_obs) & 
                         (data['LIGNE'] == ligne) & 
                         (data['KILOMETRAGE_CLASSE'] == kilometrage_classe)]
     # Vérifier si des données sont disponibles après le filtrage
    if not filtered_data.empty:
        st.header("Détails du véhicule sélectionné")
        st.write(filtered_data)

        # Préparer les données pour train_and_evaluate_model_tree
        A_pred = {
            'MODELE': modele,
            'CONSTRUCTEUR': constructeur,
            'MOTEUR': moteur,
            'SIG_ORGANE': sig_organe,
            'SIG_OBS': sig_obs,
            'LIGNE': ligne,
            'KILOMETRAGE_CLASSE': kilometrage_classe
        }

        # Appeler la fonction train_and_evaluate_model_tree avec les données de l'utilisateur
        accuracy, max_proba_class, max_proba_value = train_and_evaluate_model_tree(data, target_variable='SYSTEM_N1', A_pred=A_pred)

        # Afficher les résultats
        st.subheader("Résultats de l'arbre de décision pour SYSTEM_N1")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Classe prédite avec la probabilité maximale: {max_proba_class}")
        st.write(f"Probabilité maximale: {max_proba_value}")
    else:
        st.error("Aucune donnée trouvée pour les sélections données. Veuillez ajuster les filtres.")

    st.markdown('</div>', unsafe_allow_html=True)

