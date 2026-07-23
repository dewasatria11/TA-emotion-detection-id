# Analisis Teks: Emotion Detection pada Teks Bahasa Indonesia

Proyek ini bertujuan untuk membangun model deteksi emosi (*emotion detection*) pada teks Bahasa Indonesia dengan membandingkan dua arsitektur transformer terpopuler:
1. **IndoBERT** (`indobenchmark/indobert-base-p1`) — Model monolingual Bahasa Indonesia.
2. **XLM-RoBERTa** (`xlm-roberta-base`) — Model multilingual.

Pendekatan fine-tuning menggunakan **Full Fine-Tuning** (bukan parameter-efficient tuning seperti LoRA) menggunakan pustaka Hugging Face `transformers.Trainer`.

---

## Ringkasan Dataset

- **Nama Dataset**: IndoNLU EmoT (`indonlp/indonlu` config `emot`)
- **Pembagian Kelas**: 5 kelas emosi (*sadness, anger, love, fear, happy*)
- **Ukuran Dataset**:
  - **Train**: 4,400 sampel
  - **Validation**: 440 sampel
  - **Test**: 440 sampel
- **Catatan**: Pembagian data (*split*) sudah disediakan langsung dari sumber dataset IndoNLU, sehingga tidak dilakukan split ulang untuk menjaga konsistensi perbandingan hasil penelitian.

---

## Model yang Dibandingkan

1. **IndoBERT-base-p1**
   - **Tipe**: Monolingual (Indonesia)
   - **Ukuran**: ~124M parameter
   - **Karakteristik**: Sangat baik untuk memahami kosakata sehari-hari (slang, singkatan) dalam Bahasa Indonesia karena dilatih pada korpus Indonesia yang sangat besar.
2. **XLM-RoBERTa-base**
   - **Tipe**: Multilingual (100+ bahasa)
   - **Ukuran**: ~270M parameter
   - **Karakteristik**: Baik untuk transfer learning multibahasa, namun memiliki ukuran representasi yang lebih besar.

---

## Struktur Folder Proyek

```
emotion-detection-id/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_finetune_indobert.ipynb
│   ├── 03_finetune_xlmr.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   └── result_emotion_detection_dewa.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── metrics.csv          (menyimpan hasil metrik evaluasi eksperimen)
│   └── figures/              (folder penyimpanan plot visualisasi)
└── report/
    └── laporan_akhir.md     (kerangka draf laporan akhir tugas akhir)
```

---

## Ringkasan Hasil Eksperimen

Hasil evaluasi pada data test untuk masing-masing model dan konfigurasi adalah sebagai berikut:

| Model | Learning Rate | Batch Size | Epochs | Accuracy (Test) | Macro F1 (Test) |
|---|---|---|---|---|---|
| **IndoBERT-base-p1 (Baseline)** | 2e-5 | 16 | 5 | 0.7318 | 0.7372 |
| **XLM-RoBERTa-base (Baseline)** | 2e-5 | 16 | 5 | 0.7205 | 0.7289 |
| **IndoBERT (Tuned)** | 1e-5 | 16 | 5 | **0.7409** | **0.7494** |
| **XLM-RoBERTa (Tuned)** | 1e-5 | 16 | 5 | 0.7341 | 0.7400 |

### Temuan Utama & Kesimpulan

1. **Keunggulan Model Monolingual**: IndoBERT (model monolingual Bahasa Indonesia) secara konsisten sedikit mengungguli XLM-RoBERTa (model multilingual) baik pada konfigurasi baseline maupun setelah tuning. Ini menunjukkan keunggulan model monolingual yang dilatih khusus pada korpus lokal dalam menangkap konteks teks informal/gaul khas media sosial (Twitter).
2. **Kinerja per Kelas**: Keunggulan IndoBERT terutama ditopang oleh kemampuannya membedakan kelas *sadness* dan *anger* dengan lebih baik. Sebaliknya, XLM-RoBERTa menunjukkan performa yang cukup baik pada kelas emosi yang lebih universal seperti *fear* dan *love*.
3. **Analisis Kesalahan**: Terdapat tumpang tindih leksikal/konteks yang tinggi antara kelas *anger* dan *sadness* di Twitter Bahasa Indonesia, sehingga menjadi sumber kesalahan klasifikasi terbesar pada kedua model.
4. **Visualisasi**: Gambar kurva training dan heatmap confusion matrix tersimpan pada direktori `results/figures/`.

