import pandas as pd
from gensim.models import Word2Vec
import os

class DataLoader:
    def __init__(self):
        self.models = self.load_word2vec_models()
        self.tfidf_data = self.load_tfidf_data()
        self.sentences = self.load_sentences()
        
    def load_word2vec_models(self):
        models_dir = "../../odev1/output/models"
        return {f.replace('.model', ''): Word2Vec.load(f"{models_dir}/{f}") 
                for f in os.listdir(models_dir) if f.endswith('.model')}
    
    def load_tfidf_data(self):
        return {
            'lemmatized': pd.read_csv("../../odev1/output/tfidf_lemmatized.csv"),
            'stemmed': pd.read_csv("../../odev1/output/tfidf_stemmed.csv")
        }
    
    def load_sentences(self):
        return {
            'lemmatized': pd.read_csv("../../odev1/output/lemmatized_sentences.csv", header=None).iloc[:, 0].tolist(),
            'stemmed': pd.read_csv("../../odev1/output/stemmed_sentences.csv", header=None).iloc[:, 0].tolist()
        } 