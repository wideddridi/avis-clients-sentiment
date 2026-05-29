# 🎬 Analyse de Sentiment — Avis Clients (IMDB)

Projet de Machine Learning pour la **classification automatique du sentiment** (positif / négatif) de critiques de films, entraîné sur le dataset IMDB 50 000 reviews.

---

## 📋 Table des matières

- [🎬 Analyse de Sentiment — Avis Clients (IMDB)](#-analyse-de-sentiment--avis-clients-imdb)
  - [📋 Table des matières](#-table-des-matières)
  - [Aperçu du projet](#aperçu-du-projet)
  - [Structure du projet](#structure-du-projet)
  - [Installation](#installation)
    - [Prérequis](#prérequis)
    - [Étapes](#étapes)
  - [Lancement](#lancement)
    - [Pipeline complet (entraînement + sauvegarde)](#pipeline-complet-entraînement--sauvegarde)
  - [Pipeline](#pipeline)
  - [Modèles](#modèles)
    - [Logistic Regression](#logistic-regression)
    - [LinearSVC](#linearsvc)
    - [Random Forest](#random-forest)
    - [XGBoost](#xgboost)
    - [MLP (Réseau de neurones)](#mlp-réseau-de-neurones)
    - [Vectorisation TF-IDF](#vectorisation-tf-idf)
  - [Résultats](#résultats)
  - [Technologies utilisées](#technologies-utilisées)
  - [Auteur](#auteur)

---

## Aperçu du projet

Ce projet implémente un **pipeline complet d'analyse de sentiment** sur des critiques de films :

- **Données** : IMDB Dataset (50 000 reviews, équilibrées 50% positives / 50% négatives)
- **Approche** : Nettoyage textuel → Lemmatisation spaCy → TF-IDF + bigrammes → Classification
- **Modèles comparés** : Logistic Regression, LinearSVC, Random Forest, XGBoost, MLP
- **Meilleur modèle** : LinearSVC (~90% F1-score)
- **Objectif** : Prédire automatiquement si une critique est **positive** ou **négative**

---

## Structure du projet

```
avis_clients_project/
│
├── backend/
│   ├── data/
│   │   └── raw/
│   │   │   └── IMDB Dataset.csv          # Dataset source (50 000 reviews)
│   │   └── clean/
│   │       └── imdb_clean.csv
│   ├── notebooks/                        # Exploration et analyse
│   │   ├── 01_exploration.ipynb          # Distribution, statistiques, 
│   │   └── 02_model.ipynb                # Graphes, matrice de confusion, ROC
│   │
│   └── src/                              # Code source réutilisable
│       ├── preprocessing.py              # Nettoyage + tokenisation + split
│       ├── vectorisation.py              # Vectorisation TF-IDF
│       ├── train.py                      # Entraînement + évaluation des modèles
│
├── models/                               # Modèles sauvegardés (généré après main.py)
│   ├── meilleur_modele.pkl
│   └── meilleur_vectorizer.pkl
│
├── main.py                               # Point d'entrée — pipeline complet
└── README.md
```

---

## Installation

### Prérequis

- Python 3.9+
- Conda

### Étapes

**1. Cloner le dépôt**

```bash
git clone https://github.com/ton-user/avis-clients-project.git
cd avis-clients-project
```

**2. Créer et activer l'environnement**

```bash
conda create -n avis_clients python=3.9
conda activate avis_clients
```

**3. Installer les dépendances**

```bash
pip install pandas scikit-learn spacy xgboost joblib
```

**4. Télécharger le modèle de langue spaCy**

```bash
python -m spacy download en_core_web_sm
```

---

## Lancement

### Pipeline complet (entraînement + sauvegarde)

```bash
conda activate avis_clients
python main.py
```

Ce que fait `main.py` dans l'ordre :

| Étape | Action | Détail |
|-------|--------|--------|
| 1 | Chargement du CSV | 50 000 reviews IMDB |
| 2 | Encodage sentiment | `positive` → 1, `negative` → 0 |
| 3 | Nettoyage textes | HTML, URLs, contractions, ponctuation |
| 4 | Tokenisation batch | Lemmatisation spaCy + suppression stop words |
| 5 | Division train/valid/test | 70% / 10% / 20% stratifié |
| 6 | Vectorisation TF-IDF | bigrammes, max_features= 34000 |
| 7 | Entraînement + comparaison | 5 modèles comparés, sélection par F1-score |
| 8 | Sauvegarde | `models/meilleur_modele.pkl` + `meilleur_vectorizer.pkl` |

---

## Pipeline

```
Texte brut
    │
    ▼
clean_text()              → lowercase, HTML, URLs, contractions, ponctuation
    │
    ▼
tokeniser_batch()         → lemmatisation spaCy + suppression stop words
                            (négations conservées : not, never, no, neither...)
    │
    ▼
division_train_test()     → 70% train / 10% valid / 20% test (stratifié)
    │
    ▼
TfidfVectorizer()         → ngram_range=(1,2), max_features variable, min_df=3
                            sublinear_tf=True → log(1 + tf)
    │
    ▼
comparer_modeles()                 → Logistic Regression, LinearSVC, Random Forest,
                            XGBoost, MLP — sélection par F1-score
    │
    ▼
Sauvegarde joblib         → models/meilleur_modele.pkl
                            models/meilleur_vectorizer.pkl
```

---

## Modèles

### Logistic Regression

```python
LogisticRegression(
    C=1.0,
    solver='saga',     # optimal pour datasets sparses (TF-IDF)
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
```

Avantages : probabilités bien calibrées, très interprétable, entraînement rapide.

### LinearSVC

```python
CalibratedClassifierCV(
    LinearSVC(C=1.0, max_iter=2000, random_state=42)
)
```

Avantages : meilleure précision brute sur les représentations TF-IDF, entraînement rapide.  
`CalibratedClassifierCV` permet d'obtenir des probabilités (`.predict_proba()`).

### Random Forest

```python
RandomForestClassifier(
    n_estimators=200, max_depth=20,
    min_samples_split=5, random_state=42, n_jobs=-1
)
```

### XGBoost

```python
XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42, tree_method='hist'
)
```

### MLP (Réseau de neurones)

```python
MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu', solver='adam',
    max_iter=50, early_stopping=True,
    validation_fraction=0.1, random_state=42
)
```


### Vectorisation TF-IDF

```python
TfidfVectorizer(
    ngram_range=(1, 2),    # unigrammes + bigrammes
    max_features=30000,    # taille maximale du vocabulaire
    min_df=3,              # ignore les termes très rares
    sublinear_tf=True,     # log(1 + tf) — lisse les fréquences élevées
    strip_accents='unicode'
)
```

---

## Résultats

| Modèle | Accuracy | F1-Score |
|--------|----------|----------|
| Logistic Regression | ~89% |  ~89% |
| LinearSVC | ~90% |  ~90% | 
| Random Forest | ~84%  | ~84% |
| XGBoost | ~86% |  ~86% |
| MLP | ~89% |  ~89% |

> Résultats obtenus sur le jeu de test (20% du dataset, soit ~10 000 reviews).  
> Le meilleur modèle est sélectionné automatiquement selon le **F1-score pondéré**.

---

## Technologies utilisées

| Librairie | Version recommandée | Usage |
|-----------|---------------------|-------|
| pandas | ≥ 1.5 | Manipulation des données |
| scikit-learn | ≥ 1.2 | Vectorisation, modèles, métriques |
| spaCy | ≥ 3.5 | Tokenisation et lemmatisation |
| xgboost | ≥ 1.7 | Classifieur XGBoost |
| joblib | ≥ 1.2 | Sérialisation des modèles |
| matplotlib | ≥ 3.5 | Visualisations |
| seaborn | ≥ 0.12 | Matrices de confusion |

---

## Auteur

Projet réalisé dans le cadre d'un apprentissage du **NLP** et du **Machine Learning** appliqué à l'analyse de sentiment.

> Dataset source : [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — Kaggle
