import os
import subprocess

def run_odev1():
    print("ODEV-1 OTOMATIK CALISTIRMA")
    print("="*30)
    
    scripts = [
        ("data_analysis.py", "Veri analizi"),
        ("preprocess.py", "On isleme"),
        ("create_tfidf.py", "TF-IDF olusturma"),
        ("train_word2vec.py", "Word2Vec egitimi"),
        ("complete_zipf_analysis.py", "Zipf analizi"),
        ("model_examples.py", "Model ornekleri")
    ]
    
    for script, desc in scripts:
        print(f"\n{desc} ({script})...")
        try:
            result = subprocess.run(['python', script], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"BASARILI: {desc} tamamlandi")
            else:
                print(f"HATA: {desc} hatasi: {result.stderr}")
        except Exception as e:
            print(f"HATA: {script} calistirilamadi: {e}")
    
    print(f"\nODEV-1 tamamlandi!")

if __name__ == "__main__":
    run_odev1() 