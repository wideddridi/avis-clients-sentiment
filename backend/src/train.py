
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve



import time



# ============================================================
# ENTRAINEMENT
# ============================================================
def get_all_models():
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, solver='saga', max_iter=1000,
            random_state=42, n_jobs=-1
        ),
        'SVM (LinearSVC)': CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42)
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1, tree_method='hist'
        ),
        'MLP (Réseau de neurones)': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu', solver='adam',
            max_iter=50, early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ),
    }
    return models

def entrainer_tous_modeles(x_train_vec, y_train):
    models = get_all_models()
    trained_models = {}
    print("=" * 55)
    print("ENTRAÎNEMENT DES MODÈLES ML")
    print("=" * 55)
    for nom, model in models.items():
        print(f"\n⏳ {nom}...")
        start = time.time()
        model.fit(x_train_vec, y_train)
        elapsed = time.time() - start
        trained_models[nom] = model
        print(f"    Terminé en {elapsed:.1f}s")
    return trained_models
# ============================================================
# EVALUATION
# ============================================================
def evaluer_modele(model, X_vec, y, nom_modele):
    """Évalue un modèle et retourne ses métriques (sans affichage)."""
    y_pred  = model.predict(X_vec)
    y_proba = model.predict_proba(X_vec)[:, 1]
    
    accuracy  = accuracy_score(y, y_pred)
    f1        = f1_score(y, y_pred, average='weighted')
    auc_score = roc_auc_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)

    print(f"\n{'='*55}")
    print(f"  {nom_modele}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {auc_score:.4f}")
    print(classification_report(y, y_pred, target_names=['Négatif', 'Positif']))

    return {
        'nom'      : nom_modele,
        'accuracy' : accuracy,
        'f1_score' : f1,
        'auc'      : auc_score,
        'y_pred'   : y_pred,
        'y_proba'  : y_proba,
        'fpr'      : fpr,
        'tpr'      : tpr,
    }




# ============================================================
# COMPARAISON
# ============================================================
def comparer_modeles(X_train_vec, X_test_vec, y_train, y_test):
    resultats = {}
    #entrainement des modèles
    models = get_all_models()

    #évaluation des modèles
    for nom_modele, model in models.items():
        print(f"\n {nom_modele}...")
        start = time.time()
        model.fit(X_train_vec, y_train)   
        print(f"    Terminé en {time.time()-start:.1f}s")     
        res = evaluer_modele(model, X_test_vec, y_test, nom_modele)
        resultats[nom_modele] = {
            'model': model,
            'y_pred': res['y_pred'],
            'y_proba'  : res['y_proba'],
            'accuracy' : res['accuracy'],
            'f1_score' : res['f1_score'],
        }
    # Meilleur modèle
    meilleur_nom = max(resultats, key=lambda k: resultats[k]['f1_score'])
    print(f"\n Meilleur modèle : {meilleur_nom} "
          f"({resultats[meilleur_nom]['f1_score']*100:.2f}%)")

    return meilleur_nom, resultats[meilleur_nom]['model']

