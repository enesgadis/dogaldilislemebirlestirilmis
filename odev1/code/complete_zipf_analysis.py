import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

class ZipfAnalyzer:
    def __init__(self):
        self.raw_data_path = "../../data/abcnews-date-text.csv"
        self.output_path = "../output/"
        
    def analyze_zipf_law(self, text_data, title, save_path):
        # Metni birleştir ve kelimelere ayır
        if isinstance(text_data, pd.Series):
            all_text = ' '.join(text_data.astype(str).str.lower())
        else:
            all_text = ' '.join(str(text).lower() for text in text_data)
        
        words = all_text.split()
        word_freq = Counter(words)
        frequencies = sorted(word_freq.values(), reverse=True)
        ranks = range(1, len(frequencies) + 1)
        
        # Grafik çiz
        plt.figure(figsize=(10, 6))
        plt.loglog(ranks, frequencies, 'b-', alpha=0.7, linewidth=2)
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title(f'Zipf Yasasi - {title}')
        plt.grid(True, alpha=0.3)
        
        # İdeal Zipf çizgisi
        ideal_zipf = [frequencies[0] / r for r in ranks]
        plt.loglog(ranks, ideal_zipf, 'r--', alpha=0.5, label='İdeal Zipf')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'total_words': len(words),
            'unique_words': len(word_freq),
            'top_words': word_freq.most_common(10),
            'frequencies': frequencies,
            'ranks': ranks
        }
    
    def analyze_raw_data(self):
        print("HAM VERI ZIPF ANALIZI")
        print("="*30)
        
        try:
            df = pd.read_csv(self.raw_data_path)
            headlines = df['headline_text'].dropna()
            
            results = self.analyze_zipf_law(
                headlines, "Ham Veri", f"{self.output_path}zipf_raw_data.png"
            )
            
            print(f"Toplam kelime: {results['total_words']:,}")
            print(f"Benzersiz kelime: {results['unique_words']:,}")
            
            print(f"\nEn sik 5 kelime:")
            for i, (word, count) in enumerate(results['top_words'][:5], 1):
                print(f"  {i}. {word}: {count:,}")
            
            self.check_zipf_compliance(results, "Ham Veri")
            return results
            
        except FileNotFoundError:
            print("HATA: Ham veri dosyasi bulunamadi!")
            return None
    
    def analyze_processed_data(self):
        print(f"\nISLENMIS VERILER ZIPF ANALIZI")
        print("="*30)
        
        results = {}
        
        # Lemmatized
        try:
            lemmatized_df = pd.read_csv(f"{self.output_path}lemmatized_sentences.csv", header=None)
            
            lemma_results = self.analyze_zipf_law(
                lemmatized_df.iloc[:, 0], "Lemmatized", f"{self.output_path}zipf_lemmatized.png"
            )
            
            print(f"\nLemmatized: {lemma_results['total_words']:,} kelime")
            self.check_zipf_compliance(lemma_results, "Lemmatized")
            results['lemmatized'] = lemma_results
            
        except FileNotFoundError:
            print("HATA: Lemmatized veri bulunamadi!")
        
        # Stemmed
        try:
            stemmed_df = pd.read_csv(f"{self.output_path}stemmed_sentences.csv", header=None)
            
            stem_results = self.analyze_zipf_law(
                stemmed_df.iloc[:, 0], "Stemmed", f"{self.output_path}zipf_stemmed.png"
            )
            
            print(f"Stemmed: {stem_results['total_words']:,} kelime")
            self.check_zipf_compliance(stem_results, "Stemmed")
            results['stemmed'] = stem_results
            
        except FileNotFoundError:
            print("HATA: Stemmed veri bulunamadi!")
        
        return results
    
    def check_zipf_compliance(self, results, data_type):
        frequencies = results['frequencies'][:100]
        ranks = list(range(1, len(frequencies) + 1))
        ideal_frequencies = [frequencies[0] / r for r in ranks]
        
        correlation = np.corrcoef(np.log(frequencies), np.log(ideal_frequencies))[0, 1]
        print(f"{data_type} Zipf uyumu: {correlation:.3f}")
        
        if correlation > 0.8:
            print("Guclu uyum")
        elif correlation > 0.6:
            print("Orta uyum")
        else:
            print("Zayif uyum")

if __name__ == "__main__":
    analyzer = ZipfAnalyzer()
    raw_results = analyzer.analyze_raw_data()
    processed_results = analyzer.analyze_processed_data()
    print(f"\nZipf analizleri tamamlandi!") 