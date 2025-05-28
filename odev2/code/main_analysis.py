from data_loader import DataLoader
from similarity_calculator_mini import SimilarityCalculator
from evaluator import Evaluator
from visualizer import Visualizer
from reporter import Reporter

def main():
    print("Mini Benzerlik Analizi Basliyor...")
    
    # Veri yukle
    loader = DataLoader()
    calc = SimilarityCalculator()
    eval = Evaluator()
    viz = Visualizer()
    rep = Reporter()
    
    # Query metinleri
    query_lem = "australia contribute million aid iraq"
    query_stem = "australia contribut million aid iraq"
    
    # Benzerlik hesapla
    results = {}
    
    # TF-IDF hesaplamalari
    results['tfidf_lemmatized'] = calc.calculate_tfidf_similarity(
        query_lem, 'lemmatized', loader.tfidf_data, loader.sentences)
    results['tfidf_stemmed'] = calc.calculate_tfidf_similarity(
        query_stem, 'stemmed', loader.tfidf_data, loader.sentences)
    
    # Word2Vec hesaplamalari
    for model_name, model in loader.models.items():
        data_type = 'lemmatized' if 'lemmatized' in model_name else 'stemmed'
        query = query_lem if data_type == 'lemmatized' else query_stem
        results[model_name] = calc.calculate_word2vec_similarity(
            query, model, loader.sentences[data_type])
    
    # Degerlendirme
    all_scores = {}
    for model_name, model_results in results.items():
        data_type = 'lemmatized' if 'lemmatized' in model_name else 'stemmed'
        scores = eval.subjective_evaluation({model_name: model_results}, loader.sentences[data_type])
        all_scores.update(scores)
    
    # Jaccard analizi
    jaccard_matrix, model_names = eval.ranking_agreement(results)
    
    # Gorsellestirme ve rapor
    viz.create_heatmap(jaccard_matrix, model_names)
    rep.generate_report(all_scores, jaccard_matrix, model_names)
    
    print("Analiz tamamlandi!")
    print(f"Toplam {len(model_names)} model analiz edildi")
    print(f"En iyi model: {max(all_scores.items(), key=lambda x: x[1])}")

if __name__ == "__main__":
    main() 