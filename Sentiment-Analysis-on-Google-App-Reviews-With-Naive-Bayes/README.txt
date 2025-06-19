# 📊 Review Sentiment Analysis App - Naive Bayes

Aplikasi berbasis web menggunakan Streamlit untuk menganalisis sentimen dari teks ulasan berbahasa Inggris atau Indonesia.
Didukung oleh model Naive Bayes, aplikasi ini mampu memprediksi apakah sentimen dari kalimat 
atau kumpulan ulasan bernada positif atau negatif.

 Fitur:

- Prediksi sentimen satu kalimat (input manual).
- Prediksi massal menggunakan file CSV.
- Pra-pemrosesan teks untuk dua bahasa: Inggris (lemmatization) & Indonesia (stemming).
- Translasi otomatis dari Bahasa Indonesia ke Bahasa Inggris.
- Visualisasi hasil prediksi:
  - Bar chart distribusi sentimen.
  - Wordcloud kata-kata paling sering muncul.
  - Pie chart distribusi presentase hasil.
- Fitur unduh hasil prediksi dalam format CSV.

 Teknologi yang Digunakan

- Python
- Streamlit
- Pandas, NumPy
- NLTK, Sastrawi
- scikit-learn
- joblib
- deep_translator
- matplotlib, wordcloud
