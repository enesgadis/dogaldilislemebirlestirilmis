from data_loader import DataLoader
from similarity_calculator_mini import SimilarityCalculator
from evaluator import Evaluator

def print_detailed_results():
    print("DETAYLI BENZERLIK SONUCLARI")
    print("=" * 60)
    
    # Veri yükle
    loader = DataLoader()
    calc = SimilarityCalculator()
    eval = Evaluator()
    
    # Query metinleri
    query_lem = "australia contribute million aid iraq"
    query_stem = "australia contribut million aid iraq"
    
    print(f"\nQuery Metni (Lemmatized): '{query_lem}'")
    print(f"Query Metni (Stemmed): '{query_stem}'")
    print("=" * 60)
    
    # Benzerlik hesapla ve sonuçları göster
    all_results = {}
    all_scores = {}
    
    # TF-IDF Sonuçları
    print("\n1. TF-IDF BENZERLIK SONUCLARI")
    print("-" * 40)
    
    for data_type in ['lemmatized', 'stemmed']:
        model_name = f'tfidf_{data_type}'
        query = query_lem if data_type == 'lemmatized' else query_stem
        results = calc.calculate_tfidf_similarity(query, data_type, loader.tfidf_data, loader.sentences)
        all_results[model_name] = results
        
        print(f"\n{model_name.upper()}:")
        for rank, (idx, score) in enumerate(results, 1):
            sentence = loader.sentences[data_type][idx]
            print(f"  {rank}. [{score:.4f}] {sentence}")
        
        # Subjective skorları
        scores = eval.subjective_evaluation({model_name: results}, loader.sentences[data_type])
        all_scores.update(scores)
        print(f"  Ortalama Subjective Skor: {scores[model_name]}/5.0")
    
    # Word2Vec Sonuçları
    print("\n\n2. WORD2VEC BENZERLIK SONUCLARI")
    print("-" * 40)
    
    for model_name, model in sorted(loader.models.items()):
        data_type = 'lemmatized' if 'lemmatized' in model_name else 'stemmed'
        query = query_lem if data_type == 'lemmatized' else query_stem
        results = calc.calculate_word2vec_similarity(query, model, loader.sentences[data_type])
        all_results[model_name] = results
        
        print(f"\n{model_name.upper()}:")
        for rank, (idx, score) in enumerate(results, 1):
            sentence = loader.sentences[data_type][idx]
            print(f"  {rank}. [{score:.4f}] {sentence}")
        
        # Subjective skorları
        scores = eval.subjective_evaluation({model_name: results}, loader.sentences[data_type])
        all_scores.update(scores)
        print(f"  Ortalama Subjective Skor: {scores[model_name]}/5.0")
    
    # Genel değerlendirme
    print("\n\n3. GENEL DEGERLENDIRME")
    print("-" * 40)
    
    print("\nEn Yüksek Skorlu Modeller:")
    top_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (model, score) in enumerate(top_models, 1):
        print(f"  {i}. {model}: {score}/5.0")
    
    print("\nModel Tipi Karsilastirmasi:")
    tfidf_avg = sum(score for model, score in all_scores.items() if 'tfidf' in model) / 2
    w2v_avg = sum(score for model, score in all_scores.items() if 'tfidf' not in model) / 16
    print(f"  TF-IDF Ortalama: {tfidf_avg:.2f}/5.0")
    print(f"  Word2Vec Ortalama: {w2v_avg:.2f}/5.0")
    
    return all_results, all_scores

if __name__ == "__main__":
    print_detailed_results() 