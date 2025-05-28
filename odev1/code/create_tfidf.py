import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np

def create_tfidf_files():
    """Create TF-IDF CSV files for both lemmatized and stemmed data"""
    
    print("TF-IDF dosyalari olusturuluyor...")
    
    # Output klasörü oluştur
    os.makedirs("output", exist_ok=True)
    
    # Veri setlerini yükle
    print("Veri setleri yukleniyor...")
    try:
        df_lem = pd.read_csv("output/lemmatized_sentences.csv", header=None)
        df_stem = pd.read_csv("output/stemmed_sentences.csv", header=None)
        print(f"Lemmatized: {len(df_lem)} satir")
        print(f"Stemmed: {len(df_stem)} satir")
    except Exception as e:
        print(f"Veri yukleme hatasi: {e}")
        return
    
    # Lemmatized için TF-IDF
    print("\nLemmatized TF-IDF hesaplaniyor...")
    vectorizer_lem = TfidfVectorizer(max_features=5000)  # Memory için sınırlı özellik
    tfidf_lem = vectorizer_lem.fit_transform(df_lem.iloc[:, 0].fillna(''))
    
    # DataFrame oluştur
    tfidf_lem_df = pd.DataFrame(
        tfidf_lem.toarray(), 
        columns=vectorizer_lem.get_feature_names_out()
    )
    tfidf_lem_df.insert(0, 'document_id', [f'doc_{i}' for i in range(len(tfidf_lem_df))])
    tfidf_lem_df.to_csv("output/tfidf_lemmatized.csv", index=False)
    print(f"tfidf_lemmatized.csv olusturuldu: {tfidf_lem_df.shape}")
    
    # Stemmed için TF-IDF  
    print("\nStemmed TF-IDF hesaplaniyor...")
    vectorizer_stem = TfidfVectorizer(max_features=5000)
    tfidf_stem = vectorizer_stem.fit_transform(df_stem.iloc[:, 0].fillna(''))
    
    # DataFrame oluştur
    tfidf_stem_df = pd.DataFrame(
        tfidf_stem.toarray(), 
        columns=vectorizer_stem.get_feature_names_out()
    )
    tfidf_stem_df.insert(0, 'document_id', [f'doc_{i}' for i in range(len(tfidf_stem_df))])
    tfidf_stem_df.to_csv("output/tfidf_stemmed.csv", index=False)
    print(f"tfidf_stemmed.csv olusturuldu: {tfidf_stem_df.shape}")
    
    print("\nTF-IDF dosyalari basariyla olusturuldu!")
    
    return vectorizer_lem, vectorizer_stem

if __name__ == "__main__":
    create_tfidf_files() 