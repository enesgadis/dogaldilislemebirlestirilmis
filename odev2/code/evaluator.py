import numpy as np
from typing import Set

class Evaluator:
    def subjective_evaluation(self, results: dict, sentences: list) -> dict:
        scores = {}
        for model_name, model_results in results.items():
            model_scores = []
            for idx, sim_score in model_results:
                sentence = sentences[idx]
                base_score = 3.0
                if 'iraq' in sentence.lower(): base_score += 1.0
                if any(word in sentence.lower() for word in ['australia', 'aid', 'million', 'contribut']): 
                    base_score += 0.5
                if sim_score > 0.8: base_score += 0.5
                elif sim_score > 0.6: base_score += 0.3
                model_scores.append(min(5.0, base_score))
            scores[model_name] = round(np.mean(model_scores), 2)
        return scores
    
    def jaccard_similarity(self, set1: Set[int], set2: Set[int]) -> float:
        return len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0.0
    
    def ranking_agreement(self, all_results: dict):
        model_names = list(all_results.keys())
        n_models = len(model_names)
        jaccard_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    jaccard_matrix[i][j] = 1.0
                else:
                    set1 = set([idx for idx, _ in all_results[model1]])
                    set2 = set([idx for idx, _ in all_results[model2]])
                    jaccard_matrix[i][j] = self.jaccard_similarity(set1, set2)
        
        return jaccard_matrix, model_names 