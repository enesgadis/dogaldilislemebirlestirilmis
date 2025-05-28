import numpy as np
from typing import List

class Reporter:
    def generate_report(self, scores: dict, jaccard_matrix: np.ndarray, model_names: List[str], 
                       save_path: str = '../output/mini_analysis_report.txt'):
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("MINI BENZERLIK ANALIZI RAPORU\n")
            f.write("="*40 + "\n\n")
            
            # En iyi modeller
            f.write("EN IYI 3 MODEL:\n")
            f.write("-"*15 + "\n")
            top_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (model, score) in enumerate(top_models, 1):
                f.write(f"{i}. {model}: {score}/5.0\n")
            
            # Model karsilastirmalari
            f.write(f"\nMODEL KARSILASTIRMALARI:\n")
            f.write("-"*20 + "\n")
            tfidf_avg = np.mean([score for model, score in scores.items() if 'tfidf' in model])
            w2v_avg = np.mean([score for model, score in scores.items() if 'tfidf' not in model])
            f.write(f"TF-IDF Ortalama: {tfidf_avg:.2f}/5.0\n")
            f.write(f"Word2Vec Ortalama: {w2v_avg:.2f}/5.0\n")
            
            # Jaccard ozeti
            avg_agreement = np.mean(jaccard_matrix[np.triu_indices_from(jaccard_matrix, k=1)])
            f.write(f"\nOrtalama Jaccard Benzerligi: {avg_agreement:.3f}\n")
        
        print(f"Rapor kaydedildi: {save_path}") 