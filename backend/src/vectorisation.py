from sklearn.feature_extraction.text import TfidfVectorizer


def vectoriser(x_train, x_valid, x_test, ngram_range=(1, 2), max_features=34000, min_df=3):
    """
    Vectorisation TF-IDF avec bigrammes.
    
    Paramètres:
    -----------
    x_train : pd.Series — textes d'entraînement (déjà tokenisés + joints en string)
    x_valid : pd.Series — textes de validation
    x_test  : pd.Series — textes de test
    ngram_range  : tuple — (1,2) = unigrammes + bigrammes
    max_features : int   — taille max du vocabulaire
    min_df       : int   — ignorer les mots présents dans moins de N documents
    
    Retourne:
    ---------
    x_train_vec, x_valid_vec, x_test_vec, vectorizer
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,   
        strip_accents='unicode',  
        analyzer='word',          
    )
    
    x_train_vec = vectorizer.fit_transform(x_train)  
    x_valid_vec = vectorizer.transform(x_valid)
    x_test_vec  = vectorizer.transform(x_test)        
    
    print(f"✅ Vectorisation effectuée:")
    print(f"   Taille du vocabulaire : {len(vectorizer.vocabulary_)}")
    print(f"   Shape train : {x_train_vec.shape}")
    print(f"   Shape valid : {x_valid_vec.shape}")
    print(f"   Shape test  : {x_test_vec.shape}")
    print(f"   Densité matrice train : {x_train_vec.nnz / (x_train_vec.shape[0] * x_train_vec.shape[1]):.4f}")
    
    return x_train_vec, x_valid_vec, x_test_vec, vectorizer