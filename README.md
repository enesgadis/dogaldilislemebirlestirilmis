#  ABC News Headlines - Doğal Dil İşleme Projesi

**Gümüşhane Üniversitesi - Doğal Dil İşleme Dersi**  
**Öğrenci:** [Enes Gadiş]  
**Öğrenci No:** [2107231053]

##  Proje Özeti

Bu proje, ABC News Headlines veri seti üzerinde kapsamlı doğal dil işleme analizi gerçekleştiren iki aşamalı bir çalışmadır:

- **ÖDEV 1:** Veri ön işleme, Word2Vec model eğitimi ve TF-IDF vektörizasyonu
- **ÖDEV 2:** Metin benzerlik analizi ve model performans değerlendirmesi

## Proje Yapısı

```
📦 abcnews_nlp_project_full/
├── 📁 data/                          # Ham veri dosyaları
│   └── abcnews-date-text.csv         # ABC News veri seti (60.85 MB)
│
├── 📁 odev1/                         # ÖDEV 1 - Veri İşleme ve Model Eğitimi
│   ├── 📁 code/                      # Kaynak kodları
│   │   ├── preprocess.py             # Veri ön işleme (tokenization, lemmatization, stemming)
│   │   ├── train_word2vec.py         # Word2Vec model eğitimi
│   │   ├── create_tfidf.py           # TF-IDF vektörizasyon
│   │   ├── zipf_analysis.py          # Zipf yasası analizi
│   │   ├── complete_zipf_analysis.py # Detaylı Zipf analizi
│   │   ├── data_analysis.py          # Veri analizi ve istatistikler
│   │   ├── model_examples.py         # Model kullanım örnekleri
│   │   ├── test_similarity.py        # Benzerlik testi
│   │   ├── tfidf_vectorizer.py       # TF-IDF araçları
│   │   └── run_all.py               # Tüm işlemleri çalıştır
│   │
│   └── 📁 output/                    # Çıktı dosyaları
│       ├── 📁 models/                # Eğitilmiş Word2Vec modelleri (16 adet)
│       │   ├── lemmatized_model_cbow_window2_dim100.model
│       │   ├── lemmatized_model_cbow_window2_dim300.model
│       │   ├── lemmatized_model_cbow_window4_dim100.model
│       │   ├── lemmatized_model_cbow_window4_dim300.model
│       │   ├── lemmatized_model_skipgram_window2_dim100.model
│       │   ├── lemmatized_model_skipgram_window2_dim300.model
│       │   ├── lemmatized_model_skipgram_window4_dim100.model
│       │   ├── lemmatized_model_skipgram_window4_dim300.model
│       │   ├── stemmed_model_cbow_window2_dim100.model
│       │   ├── stemmed_model_cbow_window2_dim300.model
│       │   ├── stemmed_model_cbow_window4_dim100.model
│       │   ├── stemmed_model_cbow_window4_dim300.model
│       │   ├── stemmed_model_skipgram_window2_dim100.model
│       │   ├── stemmed_model_skipgram_window2_dim300.model
│       │   ├── stemmed_model_skipgram_window4_dim100.model
│       │   └── stemmed_model_skipgram_window4_dim300.model
│       │
│       ├── 📁 plots/                 # Görselleştirmeler
│       │   ├── zipf_original.png     # Ham veri Zipf grafiği
│       │   ├── zipf_lemmatized.png   # Lemmatized veri Zipf grafiği
│       │   └── zipf_stemmed.png      # Stemmed veri Zipf grafiği
│       │
│       ├── 📁 analysis/              # Analiz raporları
│       ├── tfidf_lemmatized.csv      # TF-IDF matrisi (lemmatized) - 191.57 MB
│       ├── tfidf_stemmed.csv         # TF-IDF matrisi (stemmed) - 191.59 MB
│       ├── lemmatized_sentences.csv  # İşlenmiş cümleler (lemmatized)
│       ├── stemmed_sentences.csv     # İşlenmiş cümleler (stemmed)
│       ├── duplicate_titles.csv      # Duplikasyonlar listesi
│       ├── duplicate_pairs.txt       # Duplikasyon çiftleri
│       └── zipf_raw_data.png         # Ham veri görselleştirmesi
│
├── 📁 odev2/                         # ÖDEV 2 - Benzerlik Analizi ve Değerlendirme
│   ├── 📁 code/                      # Kaynak kodları
│   │   ├── main_analysis.py          # Ana analiz motoru
│   │   ├── comprehensive_evaluation.py # Kapsamlı değerlendirme sistemi
│   │   ├── similarity_calculator_mini.py # Benzerlik hesaplayıcı
│   │   ├── detailed_similarity_results.py # Detaylı sonuç analizi
│   │   ├── data_loader.py            # Veri yükleme araçları
│   │   ├── evaluator.py              # Değerlendirme metrikleri
│   │   ├── visualizer.py             # Görselleştirme araçları
│   │   ├── reporter.py               # Rapor üretici
│   │   └── run_all.py               # Tüm analizleri çalıştır
│   │
│   └── 📁 output/                    # Analiz sonuçları
│       ├── jaccard_similarity_matrix.csv     # Jaccard benzerlik matrisi
│       ├── jaccard_similarity_heatmap.png    # Benzerlik ısı haritası (1.0 MB)
│       ├── jaccard_heatmap_mini.png          # Kompakt ısı haritası
│       └── mini_analysis_report.txt          # Analiz raporu
│
├── 📁 venv/                          # Python sanal ortamı
├── 📄 requirements.txt               # Gerekli Python paketleri
├── 📄 .gitignore                     # Git yok sayma kuralları
└── 📄 README.md                      # Bu dosya
```

## Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install -r requirements.txt
```

**Ana Paketler:**
- `gensim==4.3.0` - Word2Vec modelleri için
- `scikit-learn==1.3.0` - TF-IDF ve makine öğrenmesi
- `pandas==2.0.3` - Veri manipülasyonu
- `numpy==1.24.3` - Sayısal hesaplamalar
- `matplotlib==3.7.2` - Görselleştirme
- `seaborn==0.12.2` - İstatistiksel görselleştirme
- `nltk==3.8.1` - Doğal dil işleme araçları

### ÖDEV 1 - Veri İşleme ve Model Eğitimi

```bash
cd odev1/code
python run_all.py  # Tüm işlemleri sırayla çalıştır
```

**Veya ayrı ayrı:**
```bash
python preprocess.py         # Veri ön işleme
python train_word2vec.py     # Word2Vec model eğitimi
python create_tfidf.py       # TF-IDF vektörizasyon
python zipf_analysis.py      # Zipf yasası analizi
```

### ÖDEV 2 - Benzerlik Analizi ve Değerlendirme


cd odev2/code
python run_all.py  # Tüm analizleri çalıştır



python main_analysis.py               # Ana benzerlik analizi
python comprehensive_evaluation.py    # Kapsamlı değerlendirme
python detailed_similarity_results.py # Detaylı sonuç analizi


##  Ödev 1 - Teknik Detaylar

### Veri Seti
- **Kaynak:** ABC News Headlines (Kaggle)
- **Boyut:** 1,103,663 haber başlığı (~60.85 MB)
- **Tarih Aralığı:** 2003-2017
- **Format:** CSV (date, headline_text)

### Veri Ön İşleme
1. **Tokenization:** Metni kelimelere ayırma
2. **Temizleme:** Noktalama işaretleri ve sayıların kaldırılması
3. **Stopword Removal:** Gereksiz kelimelerin filtrelenmesi
4. **Lemmatization:** Kelimelerin kök hallerine çevrilmesi
5. **Stemming:** Kelimelerin kök hallerine çevrilmesi (alternatif)

### Word2Vec Modelleri (16 Adet)
**Parametreler:**
- **Algoritmalar:** CBOW, Skip-gram
- **Vektör Boyutları:** 100, 300
- **Pencere Boyutları:** 2, 4
- **Veri Türleri:** Lemmatized, Stemmed

### TF-IDF Vektörizasyon
- **Vocabulary Size:** ~50,000 unique terms
- **Matrix Dimensions:** 1,103,663 x ~50,000
- **Output Files:** 191+ MB per matrix

### Zipf Yasası Analizi
- Ham veri, lemmatized ve stemmed veriler için frekans analizi
- Görselleştirmeler ve istatistiksel değerlendirmeler

## Ödev 2 - Benzerlik Analizi

### Test Sorgusu
**Seçilen Metin:** "Australia contribute million aid Iraq"
- **Lemmatized:** "australia contribute million aid iraq"
- **Stemmed:** "australia contribut million aid iraq"

### Benzerlik Metrikleri
1. **Cosine Similarity:** TF-IDF ve Word2Vec vektörleri için
2. **Jaccard Similarity:** Model sonuçları arası uyum analizi

### Değerlendirme Sistemi
1. **Subjective Evaluation:** 1-5 puan skalası ile manuel değerlendirme
2. **Ranking Agreement:** Modeller arası sıralama uyumu analizi
3. **Performance Comparison:** Model türleri arası karşılaştırma

##  Ana Bulgular

### Model Performansı
- **En İyi Modeller:** Lemmatized CBOW modelleri (ortalama 4.80/5.00)
- **Word2Vec vs TF-IDF:** Word2Vec modelleri TF-IDF'ten daha iyi performans (4.40-4.80 vs 3.20)
- **CBOW vs Skip-gram:** CBOW modelleri daha tutarlı sonuçlar
- **Lemmatization vs Stemming:** Lemmatization genel olarak daha iyi

### Ranking Agreement
- CBOW modelleri arasında mükemmel uyum (1.0)
- Skip-gram modelleri arasında yüksek uyum (0.67-1.0)
- TF-IDF ve Word2Vec arasında düşük uyum

## Teknik Notlar

### Büyük Dosyalar
Büyük dosyalar `.gitignore` ile hariç tutulmuştur:
- Model dosyaları (*.model)
- TF-IDF matrisleri (*.csv)
- İşlenmiş veri dosyaları
- Ham veri seti

### Sistem Gereksinimleri
- **RAM:** Minimum 8GB (16GB önerilen)
- **Disk:** ~2GB boş alan
- **Python:** 3.8+ 

##  Raporlama

Her ödev için detaylı raporlar `output/` klasörlerinde bulunmaktadır:
- Sayısal sonuçlar (CSV)
- Görselleştirmeler (PNG)
- Analiz raporları (TXT)

##  Katkıda Bulunanlar

**Geliştirici:** [Enes Gadiş]  
**Ders:** Doğal Dil İşleme  
**Kurum:** Gümüşhane Üniversitesi  
**Yıl:** 2025

---

*Bu proje akademik amaçlar için geliştirilmiştir.*
