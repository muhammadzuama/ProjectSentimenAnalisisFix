# Sentiment Analisis Pilpres 2024

Proyek ini melakukan analisis sentimen terkait Pemilihan Presiden (Pilpres) 2024 menggunakan beberapa model machine learning. Model dengan performa terbaik adalah **Naive Bayes** dengan metrik evaluasi sebagai berikut:

- **Accuracy:** 0.6419  
- **Precision:** 0.6509  
- **Recall:** 0.6457  
- **F1 Score:** 0.6417  

---


````markdown
## Instalasi Library yang Dibutuhkan

Pastikan Anda berada di folder proyek, lalu jalankan perintah berikut untuk memasang semua dependensi yang diperlukan:
pip install -r requirements.txt
````

Jika Anda belum memiliki file `requirements.txt`, buatlah file tersebut dengan isi sebagai berikut:

```
streamlit==1.45.0
pandas==2.2.3
numpy==1.24.3
tensorflow==2.13.0
joblib==1.3.1
scikit-learn==1.6.1
```

---

## Versi Software

* TensorFlow version: 2.13.0
* Joblib version: 1.3.1
* Scikit-learn version: 1.6.1
* Python version: 3.11.3 

---

## Cara Menjalankan Aplikasi Streamlit

1. Pastikan semua library sudah terinstal.
2. Jalankan aplikasi dengan perintah:

```bash
streamlit run app.py
```

3. Akses aplikasi melalui browser pada alamat yang muncul di terminal (biasanya [http://localhost:8501](http://localhost:8501)).

---
