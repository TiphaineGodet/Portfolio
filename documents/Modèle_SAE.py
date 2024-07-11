import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
import streamlit as st
import warnings
import numpy as np


warnings.filterwarnings('ignore')

# Fonction pour charger les données et entraîner les modèles
def main():
    # Chargement des fichiers de données
    ot_odr_filename = os.path.join(".", "OT_ODR.csv.bz2")
    ot_odr_df = pd.read_csv(ot_odr_filename, compression="bz2", sep=";")

    equipements_filename = os.path.join(".", 'EQUIPEMENTS.csv')
    equipements_df = pd.read_csv(equipements_filename, sep=";")

    # Définition des variables catégorielles
    var_sig = ["SIG_ORGANE", "SIG_CONTEXTE", "SIG_OBS"]
    var_sys = ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"]
    var_odr = ["TYPE_TRAVAIL", "ODR_LIBELLE"]
    var_cat = var_sig + var_sys + var_odr

    # Conversion en variables catégorielles
    for var in var_cat:
        ot_odr_df[var] = ot_odr_df[var].astype('category')

    # Jointure des données
    data = pd.merge(ot_odr_df, equipements_df, on='EQU_ID')

    # Définition des intervalles de kilométrage
    bins = range(0, int(data["KILOMETRAGE"].max()) + 50000, 50000)
    labels = [f'{i}-{i+49999}' for i in bins[:-1]]
    data['KILOMETRAGE_CLASSE'] = pd.cut(data["KILOMETRAGE"], bins=bins, labels=labels, include_lowest=True)

    # Création de dummies pour SIG_CONTEXTE
    dummies = data['SIG_CONTEXTE'].str.get_dummies(sep='/')
    data_origine = pd.concat([data, dummies], axis=1)

    # Exécution des modèles et affichage des résultats avec Plotly
    accuracies_nb = []
    for target in ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"]:
        accuracy_nb = train_and_evaluate_model(data, target, model_type='nb')
        accuracies_nb.append(accuracy_nb)

    accuracies_tree = []
    for target in ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"]:
        accuracy_tree = train_and_evaluate_model(data, target, model_type='tree')
        accuracies_tree.append(accuracy_tree)

    accuracies_mlp = []
    for target in ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"]:
        accuracy_mlp = train_and_evaluate_model(data, target, model_type='mlp')
        accuracies_mlp.append(accuracy_mlp)

    # Génération du graphique Plotly
    fig = generate_plotly_chart(accuracies_nb, accuracies_tree, accuracies_mlp)

    # Retourne la figure Plotly
    return fig

# Fonction pour générer le graphique Plotly
def generate_plotly_chart(accuracies_nb, accuracies_tree, accuracies_mlp):
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Naive Bayes', x=["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"], y=accuracies_nb))
    fig.add_trace(go.Bar(name='Decision Tree', x=["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"], y=accuracies_tree))
    fig.add_trace(go.Bar(name='MLP', x=["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"], y=accuracies_mlp))

    fig.update_layout(title='Accuracies pour SYSTEM_N1, SYSTEM_N2, et SYSTEM_N3',
                      xaxis_title='Variable cible',
                      yaxis_title='Accuracy',
                      barmode='group')

    return fig

# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate_model(data, target_variable, model_type='nb'):
    # Liste des colonnes à supprimer
    columns_to_drop = ["DUREE_TRAVAIL", "TYPE_TRAVAIL", "SIG_CONTEXTE", "ODR_LIBELLE", "ODR_ID", "OT_ID", 
                       "EQU_ID", "KILOMETRAGE", "DATE_OT"]
    columns_to_drop += [col for col in ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"] if col != target_variable]

    # Préparation des données
    data_target = data.drop(columns_to_drop, axis=1)
    
    # Encodage des variables utiles
    le = LabelEncoder()
    for column in data_target.columns:
        data_target[column] = le.fit_transform(data_target[column])
    
    # Séparer les variables explicatives (X) et la variable à prédire (y)
    X = data_target.drop([target_variable], axis=1)
    y = data_target[target_variable]

    # Séparer les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

    # Entraînement et évaluation du modèle
    if model_type == 'nb':
        model = GaussianNB()
    elif model_type == 'tree':
        model = DecisionTreeClassifier(max_depth=14)
    elif model_type == 'mlp':
        model = MLPClassifier(random_state=1, hidden_layer_sizes=(20,), activation='relu', max_iter=20)
    else:
        raise ValueError("Unsupported model type. Choose 'nb' for Naive Bayes, 'tree' for Decision Tree, or 'mlp' for MLP.")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nModèle pour {target_variable} (Accuracy): {accuracy}')

    return accuracy

def train_and_evaluate_model_tree(data, target_variable, A_pred):
    # Liste des colonnes à supprimer
    columns_to_drop = ["DUREE_TRAVAIL", "TYPE_TRAVAIL", "SIG_CONTEXTE", "ODR_ID", "OT_ID",
                       "EQU_ID", "KILOMETRAGE", "DATE_OT"]
    columns_to_drop += [col for col in ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3","ODR_LIBELLE"] if col != target_variable]

    # Préparation des données
    data_target = data.drop(columns_to_drop, axis=1)

    # Séparer les variables explicatives (X) et la variable à prédire (y)
    X = data_target.drop([target_variable], axis=1)
    y = data_target[target_variable]

    # Séparer les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

    # Encodage des variables utiles sur l'ensemble d'entraînement
    label_encoders = {}
    for column in x_train.columns:
        le = LabelEncoder()
        x_train[column] = le.fit_transform(x_train[column])
        # Encoder les valeurs de x_test qui sont aussi présentes dans x_train
        x_test[column] = x_test[column].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        x_test[column] = le.transform(x_test[column])
        label_encoders[column] = le

    # Encodage de la variable cible sur l'ensemble d'entraînement
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = y_test.map(lambda s: '<unknown>' if s not in le_target.classes_ else s)
    le_target.classes_ = np.append(le_target.classes_, '<unknown>')
    y_test = le_target.transform(y_test)

    # Entraînement du modèle
    tree = DecisionTreeClassifier(max_depth=14)
    tree.fit(x_train, y_train)

    # Prédictions
    y_pred = tree.predict(x_test)
    y_pred_proba = tree.predict_proba(x_test)

    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)

    # Prédiction avec de nouvelles données (A_pred)
    A_pred = pd.DataFrame([A_pred], columns=x_train.columns)

    for column in A_pred.columns:
        if column in label_encoders:
            le = label_encoders[column]
            A_pred[column] = A_pred[column].map(lambda s: '<unknown>' if s not in le.classes_ else s)
            A_pred[column] = le.transform(A_pred[column])
   
    pred_proba = tree.predict_proba(A_pred)[0]  # Probabilités pour la première ligne de A_pred

    # Obtenir la classe prédite avec la probabilité maximale
    max_proba_index = np.argmax(pred_proba)
    max_proba_class = le_target.inverse_transform([max_proba_index])[0]
    max_proba_value = np.max(pred_proba)

    return accuracy, max_proba_class, max_proba_value

# Appel de la fonction principale pour exécuter le script
if __name__ == "__main__":
    main()
