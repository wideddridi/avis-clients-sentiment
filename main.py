import os

from backend.src.preprocessing import encode_sentiment, clean_text, tokeniser_batch,division_train_test
from backend.src.vectorisation import vectoriser
from backend.src.train import comparer_modeles

import pandas as pd




#============================================================
#Téléchargement du dataset
#============================================================
print("\n" + "="*55)
print("  ÉTAPE 1 — Chargement des données")
print("="*55)

imdb = pd.read_csv('backend/data/raw/IMDB Dataset.csv')
print(f"Dataset chargé : {imdb.shape}")


#============================================================
#copier du dataset et encodage de sentiment
#============================================================
print("\n" + "="*55)
print("  ÉTAPE 2 — Encodage des sentiments")
print("="*55)


imdb_copy = encode_sentiment(imdb, column='sentiment')

#============================================================
#Nettoyage du texte
#============================================================
print("\n" + "="*55)
print("  ÉTAPE 3 — Nettoyage des textes")
print("="*55)


imdb_copy['clean_review'] = imdb_copy['review'].apply(clean_text)

#============================================================
#Tokenisation
#============================================================
print("\n" + "="*55)
print("  ÉTAPE 4 — Tokenisation")
print("="*55)


imdb_copy = tokeniser_batch(imdb_copy, column='clean_review')

#============================================================
#Division en train/test 
#============================================================
print("\n" + "="*55)
print("  ÉTAPE 5 — Division en train/test")
print("="*55)
x_train, x_valid, x_test, y_train, y_valid, y_test = division_train_test(imdb_copy, text_col='tokenized_clean_review', target_col='sentiment', test_size=0.2, random_state=42)

#============================================================
#Vectorisation et comparaison des modèles
#============================================================
print("\n" + "="*55)
print("  ÉTAPE 6 — Vectorisation et comparaison des modèles")
print("="*55)


print(f"\n Vectorisation...")
x_train_str = x_train.apply(' '.join)
x_valid_str = x_valid.apply(' '.join)
x_test_str  = x_test.apply(' '.join)

#vectorisation
X_train_vect, X_valid_vect, X_test_vect, vectorizer = vectoriser(
    x_train_str, x_valid_str, x_test_str, max_features=34000
)
#Entrainement et évaluation des modèles
meilleur_nom, meilleur_model = comparer_modeles(X_train_vect, X_test_vect, y_train, y_test)

# ============================================================
#Sauvegarde du meilleur modèle et vectorizer
# ============================================================
print("\n" + "="*55)
print("  ÉTAPE 7 — Sauvegarde")
print("="*55)

import joblib

os.makedirs('models', exist_ok=True)
joblib.dump(meilleur_model,      'models/meilleur_modele.pkl')
joblib.dump(vectorizer, 'models/meilleur_vectorizer.pkl')
print(f"  Modèle {meilleur_nom} sauvegardé dans models/")
