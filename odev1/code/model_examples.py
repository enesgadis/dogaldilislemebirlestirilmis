from gensim.models import Word2Vec
import os

class ModelExamples:
    def __init__(self):
        self.models_dir = "../output/models/"
        
    def show_examples(self):
        print("WORD2VEC MODEL ORNEKLERI")
        print("="*40)
        
        # Test kelimesi
        test_word = "australia"
        
        for model_file in os.listdir(self.models_dir):
            if model_file.endswith('.model'):
                model_name = model_file.replace('.model', '')
                model = Word2Vec.load(f"{self.models_dir}{model_file}")
                
                print(f"\n{model_name}:")
                try:
                    similar_words = model.wv.most_similar(test_word, topn=5)
                    for word, score in similar_words:
                        print(f"   {word}: {score:.3f}")
                except KeyError:
                    print(f"   '{test_word}' kelimesi modelde bulunamadi")

if __name__ == "__main__":
    examples = ModelExamples()
    examples.show_examples() 