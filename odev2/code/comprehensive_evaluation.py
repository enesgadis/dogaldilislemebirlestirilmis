import pandas as pd
import numpy as np
from data_loader import DataLoader
from similarity_calculator_mini import SimilarityCalculator
from evaluator import Evaluator

class ComprehensiveEvaluator:
    def __init__(self):
        self.loader = DataLoader()
        self.calc = SimilarityCalculator()
        self.evaluator = Evaluator()
        
    def run_full_evaluation(self):
        print("KAPSAMLI MODEL DEGERLENDIRMESI")
        print("="*50)
        
        # Query metinleri
        query_lem = "australia contribute million aid iraq"
        query_stem = "australia contribut million aid iraq"
        
        all_results = {}
        
        # TF-IDF sonuçları
        for data_type in ['lemmatized', 'stemmed']:
            model_name = f'tfidf_{data_type}'
            query = query_lem if data_type == 'lemmatized' else query_stem
            results = self.calc.calculate_tfidf_similarity(
                query, data_type, self.loader.tfidf_data, self.loader.sentences
            )
            all_results[model_name] = results
        
        # Word2Vec sonuçları
        for model_name, model in self.loader.models.items():
            data_type = 'lemmatized' if 'lemmatized' in model_name else 'stemmed'
            query = query_lem if data_type == 'lemmatized' else query_stem
            results = self.calc.calculate_word2vec_similarity(
                query, model, self.loader.sentences[data_type]
            )
            all_results[model_name] = results
        
        # Subjective değerlendirme
        scores = {}
        for model_name, results in all_results.items():
            data_type = 'lemmatized' if 'lemmatized' in model_name else 'stemmed'
            model_scores = self.evaluator.subjective_evaluation(
                {model_name: results}, self.loader.sentences[data_type]
            )
            scores.update(model_scores)
        
        # Jaccard analizi
        jaccard_matrix, model_names = self.evaluator.ranking_agreement(all_results)
        
        self.print_detailed_analysis(scores, jaccard_matrix, model_names)
        
        return scores, jaccard_matrix, model_names
    
    def print_detailed_analysis(self, scores, jaccard_matrix, model_names):
        print(f"\nDETAYLI ANALIZ SONUCLARI:")
        print("-"*30)
        
        # En iyi modeller
        print("En Iyi 5 Model:")
        top_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (model, score) in enumerate(top_models, 1):
            print(f"  {i}. {model}: {score}/5.0")
        
        # Model tipi karşılaştırması
        tfidf_scores = [score for model, score in scores.items() if 'tfidf' in model]
        w2v_scores = [score for model, score in scores.items() if 'tfidf' not in model]
        
        print(f"\nModel Tipi Karsilastirmasi:")
        print(f"  TF-IDF Ortalama: {np.mean(tfidf_scores):.2f}")
        print(f"  Word2Vec Ortalama: {np.mean(w2v_scores):.2f}")
        
        # CBOW vs Skip-gram
        cbow_scores = [score for model, score in scores.items() if 'cbow' in model]
        skipgram_scores = [score for model, score in scores.items() if 'skipgram' in model]
        
        if cbow_scores and skipgram_scores:
            print(f"\nModel Mimarisi Karsilastirmasi:")
            print(f"  CBOW Ortalama: {np.mean(cbow_scores):.2f}")
            print(f"  Skip-gram Ortalama: {np.mean(skipgram_scores):.2f}")
        
        # Jaccard özet
        avg_jaccard = np.mean(jaccard_matrix[np.triu_indices_from(jaccard_matrix, k=1)])
        print(f"\nOrtalama Model Tutarliligi: {avg_jaccard:.3f}")

if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    evaluator.run_full_evaluation() 