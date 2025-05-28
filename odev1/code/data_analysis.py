import pandas as pd
import os

class DataAnalyzer:
    def __init__(self):
        self.data_path = "../../data/abcnews-date-text.csv"
        self.output_path = "../output/"
        
    def analyze_raw_data(self):
        print("ABC NEWS VERI SETI ANALIZI")
        print("="*40)
        
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print("HATA: Veri dosyasi bulunamadi!")
            return
        
        # Temel bilgiler
        file_size = os.path.getsize(self.data_path) / (1024*1024)
        print(f"Dosya: {file_size:.1f} MB, {len(df):,} satir")
        print(f"Tarih: {df['publish_date'].min()} - {df['publish_date'].max()}")
        
        # İstatistikler
        headline_lengths = df['headline_text'].str.len()
        word_counts = df['headline_text'].str.split().str.len()
        print(f"Ortalama: {headline_lengths.mean():.1f} karakter, {word_counts.mean():.1f} kelime")
        
        # Örnekler
        print(f"\nIlk 3 ornek:")
        for i, row in df.head(3).iterrows():
            print(f"  {i+1}. {row['headline_text']}")
        
        # En sık kelimeler
        all_text = ' '.join(df['headline_text'].astype(str))
        words = pd.Series(all_text.lower().split()).value_counts()
        print(f"\nEn sik 5 kelime:")
        for i, (word, count) in enumerate(words.head(5).items(), 1):
            print(f"  {i}. {word}: {count:,}")
        
        return df
    
    def compare_processed_data(self):
        print(f"\nIslenmis veri karsilastirmasi:")
        try:
            lemmatized = pd.read_csv(f"{self.output_path}lemmatized_sentences.csv", header=None)
            stemmed = pd.read_csv(f"{self.output_path}stemmed_sentences.csv", header=None)
            
            print(f"Lemmatized: {len(lemmatized):,} satir")
            print(f"Stemmed: {len(stemmed):,} satir")
            
        except FileNotFoundError:
            print("HATA: Islenmis veri bulunamadi! Once preprocess.py calistirin.")

if __name__ == "__main__":
    analyzer = DataAnalyzer()
    analyzer.analyze_raw_data()
    analyzer.compare_processed_data() 