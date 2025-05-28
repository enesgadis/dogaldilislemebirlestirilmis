import os
import subprocess

def run_odev2():
    print("ODEV-2 OTOMATIK CALISTIRMA")
    print("="*30)
    
    scripts = [
        ("main_analysis.py", "Ana analiz"),
        ("comprehensive_evaluation.py", "Kapsamli degerlendirme"),
        ("detailed_similarity_results.py", "Detayli sonuclar")
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
    
    print(f"\nODEV-2 tamamlandi!")

if __name__ == "__main__":
    run_odev2() 