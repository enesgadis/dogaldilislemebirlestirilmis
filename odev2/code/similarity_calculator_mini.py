import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

class SimilarityCalculator:
    def get_sentence_vector(self, sentence: str, model: Word2Vec) -> np.ndarray:
        vectors = [model.wv[word] for word in sentence.split() if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.wv.vector_size)
    
    def calculate_tfidf_similarity(self, query: str, data_type: str, tfidf_data: dict, sentences: dict):
        query_idx = sentences[data_type].index(query)
        tfidf_matrix = tfidf_data[data_type].iloc[:, 1:].values
        query_vector = tfidf_matrix[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        results = sorted([(i, sim) for i, sim in enumerate(similarities) if i != query_idx], 
                        key=lambda x: x[1], reverse=True)[:5]
        return results
    
    def calculate_word2vec_similarity(self, query: str, model: Word2Vec, sentences: list):
        query_idx = sentences.index(query)
        query_vector = self.get_sentence_vector(query, model)
        similarities = []
        
        for i, sentence in enumerate(sentences):
            if i != query_idx:
                sentence_vector = self.get_sentence_vector(sentence, model)
                if np.linalg.norm(query_vector) > 0 and np.linalg.norm(sentence_vector) > 0:
                    sim = np.dot(query_vector, sentence_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(sentence_vector))
                    similarities.append((i, sim))
                else:
                    similarities.append((i, 0.0))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:5] 