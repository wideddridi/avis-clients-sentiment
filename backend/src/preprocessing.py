import pandas as pd
import re
from sklearn.model_selection import train_test_split
import spacy
from spacy.lang.en.stop_words import STOP_WORDS




nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

stop_words_perso = set(STOP_WORDS)  # Utiliser les stop words de spaCy
print("i" in STOP_WORDS )  # Afficher si les 10 premiers mots sont dans STOP_WORDS

#liste de mots de négation à garder
mots_a_garder = ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor']

# On enlève les mots de négation des stop words
for mot in mots_a_garder:
    if mot in stop_words_perso:
        stop_words_perso.remove(mot)


# ============================================================
# ENCODAGE
# ============================================================
def encode_sentiment(df, column='sentiment'):
    """positive → 1, negative → 0"""
    imdb_copy = df.copy()
    imdb_copy[column] = imdb_copy[column].map({'negative': 0, 'positive': 1})
    return imdb_copy

# ============================================================
# NETTOYAGE
# ============================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()  # Convert to lowercase
    
    
    # 3. Gérer les contractions anglaises (TRÈS IMPORTANT pour le sentiment!)
    contractions = {
        "won't": "will not", "can't": "cannot", "don't": "do not",
        "it's": "it is", "i'm": "i am", "you're": "you are",
        "they're": "they are", "we're": "we are", "that's": "that is",
        "what's": "what is", "who's": "who is", "there's": "there is",
        "here's": "here is", "he's": "he is", "she's": "she is",
        "how's": "how is", "where's": "where is", "when's": "when is",
        # ORDRE IMPORTANT : les plus spécifiques d'abord
        "n't": " not", "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am",
    }
    
    for contraction, expansion in contractions.items():
        #je supprime ici r'\b'car Dans wouldn't, le ' n’est pas considéré comme un caractère de mot par Python
        text = re.sub( contraction, expansion, text)

    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs

    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'\b\d+\b', '', text)  # Supprimer les chiffres isolés (garder les mots comme "007")
    # On supprime le reste: @ # $ % ^ & * + = { } [ ] \ | / : ; < > , 
    text = re.sub(r'[,.!":;\(\)\[\]{}<>/\\|`~@#$%^&*+=]', ' ', text)
    text = re.sub(r'-+', ' ', text)  # Un ou plusieurs tirets deviennent un espace
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace


    return text

# ============================================================
# TOKENISATION
# ============================================================
def tokenizer_une_review(doc):
    tokens_final = []
    for token in doc:
        token_lemme = token.lemma_.lower()
        if token_lemme in mots_a_garder:           
            tokens_final.append(token_lemme)
        elif token_lemme not in stop_words_perso:
            tokens_final.append(token_lemme)
    return tokens_final


def tokeniser_batch(df, column='review_clean', batch_size=500):
    """
    Tokenise + lemmatise + supprime stop words sur tout le corpus en batch.
    """
    resultats = []
    texts = df[column].tolist()
    total = len(texts)
    print(f"Tokenisation de {total} reviews...")
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        resultats.append(tokenizer_une_review(doc))
        if (i + 1) % 10000 == 0:
            print(f"   {i+1}/{total} reviews traitées...")

    df['tokenized_' + column] = resultats
    print("Tokenisation terminée")
    return df


# ============================================================
# DIVISION TRAIN / TEST
# ============================================================
def division_train_test(df, text_col='tokenized_review_clean', target_col='sentiment', 
                        test_size=0.2, valid_size=0.1, random_state=42):
    """
    Divise le dataset en 3 parties : train (70%) / valid (10%) / test (20%)
    
    La validation sert à choisir le meilleur max_features.
    Le test sert à l'évaluation finale — touché une seule fois à la fin.
    """
    x = df[text_col]
    y = df[target_col]   
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )    # Étape 2 : séparer valid (10%) du reste (70%)
    # valid_size relatif au temp = 0.1 / 0.8 = 0.125
    valid_size_relative = valid_size / (1 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_temp, y_temp,
        test_size=valid_size_relative,
        random_state=random_state,
        stratify=y_temp
    )
    print(f"✅ Division effectuée:")
    print(f"   Train: {len(x_train)} échantillons")
    print(f"   Valid: {len(x_valid)} échantillons")
    print(f"   Test: {len(x_test)} échantillons")
    print(f"   Proportion positif en train: {y_train.mean():.2f}")
    print(f"   Proportion positif en valid: {y_valid.mean():.2f}")
    print(f"   Proportion positif en test: {y_test.mean():.2f}")
    return x_train, x_valid, x_test, y_train, y_valid, y_test