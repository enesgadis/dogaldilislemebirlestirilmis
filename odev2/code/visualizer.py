import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

class Visualizer:
    def create_heatmap(self, jaccard_matrix: np.ndarray, model_names: List[str], save_path: str = '../output/jaccard_heatmap_mini.png'):
        plt.figure(figsize=(12, 10))
        sns.heatmap(jaccard_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=model_names, yticklabels=model_names)
        plt.title('Model Ranking Agreement (Jaccard Similarity)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap kaydedildi: {save_path}") 