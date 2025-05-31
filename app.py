import streamlit as st
import re 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.preprocessing import LabelEncoder
# Setelah itu, fit_transform
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocessing(text):
    # case folding
    text = text.lower()

    # remove punctuation and non-alphabetic characters
    text = re.sub(r'[^\w\s]', '', text)

    # Menghapus link menggunakan regex
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
    # Menghapus hashtag menggunakan regex
    text = re.sub(r'#\S+', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # stopword removal
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    words = text.split()
    text = " ".join([word for word in words if word not in stopwords])

    return text

# Sidebar sebagai navigasi halaman
menu = st.sidebar.selectbox(
    "Pilih Halaman",
    ["General", "EDA", "Prepocessing dan Training", "Prediksi Sentimen","Kesimpulan"]
)

# Halaman General
if menu == "General":
    st.title("Selamat Datang di Aplikasi Sentimen Analisis")
    st.write("Ini adalah halaman utama aplikasi sentimen analisis.")
    st.subheader("Latar Belakang")
    st.markdown(''' 
    <div style="text-align: justify;">
    Pemilihan Umum Presiden merupakan salah satu momen krusial dalam sistem demokrasi Indonesia yang melibatkan partisipasi luas masyarakat. Di era digital saat ini, Twitter menjadi salah satu platform utama bagi masyarakat untuk menyampaikan opini politik secara real-time. Melalui cuitan-cuitan tersebut, dapat tergambarkan dinamika dukungan dan persepsi publik terhadap calon presiden. Oleh karena itu, penting dilakukan pengukuran opini publik secara kuantitatif guna memahami arah kecenderungan politik masyarakat. Dalam konteks ini, analisis sentimen menjadi alat strategis yang dapat digunakan untuk mengidentifikasi pola, opini dominan, serta persepsi publik dalam ruang digital selama masa kampanye dan pelaksanaan pemilu.
    </div>
    ''', unsafe_allow_html=True)
    st.subheader("Tujuan")
    st.markdown('''
    <div style="text-align: justify;">
    Tujuan dari aplikasi ini adalah untuk melakukan analisis sentimen terhadap cuitan-cuitan di Twitter terkait calon presiden. Dengan menggunakan teknik machine learning, aplikasi ini dapat mengklasifikasikan sentimen cuitan sebagai positif, negatif, atau netral. Hasil analisis ini diharapkan dapat memberikan wawasan yang lebih dalam mengenai opini publik terhadap calon presiden, serta membantu dalam memahami dinamika politik yang terjadi di masyarakat.
    </div>
    ''', unsafe_allow_html=True)
    st.subheader("Metode Yang Digunakan")
    st.markdown('''
    <div style="text-align: justify;">
    Metode yang digunakan dalam aplikasi ini meliputi:
    <ul>
        <li><strong>Pengumpulan Data:</strong> Mengumpulkan cuitan-cuitan di Twitter terkait calon presiden.</li>
        <li><strong>Preprocessing Data:</strong> Membersihkan dan mempersiapkan data untuk analisis, termasuk penghapusan noise, tokenisasi, dan normalisasi teks.</li>
        <li><strong>Modeling:</strong> Menggunakan algoritma machine learning seperti Naive Bayes untuk mengklasifikasikan sentimen cuitan.</li>
        <li><strong>Evaluasi Model:</strong> Mengukur akurasi model menggunakan metrik seperti confusion matrix dan classification report.</li>
        <li><strong>Visualisasi Hasil:</strong> Menyajikan hasil analisis sentimen dalam bentuk visualisasi yang mudah dipahami.</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('''
    <div style="text-align: justify;">
    Model yang digunakan dalam aplikasi ini meliputi:
    <ul>
        <li><strong>Naive Bayes:</strong> Model probabilistik sederhana yang digunakan untuk klasifikasi teks.</li>
        <li><strong>SVM:</strong> Support Vector Machine, algoritma yang efektif untuk klasifikasi dengan margin optimal.</li>
        <li><strong>GRU:</strong> Gated Recurrent Unit, jenis RNN yang digunakan untuk memproses data urutan seperti teks.</li>
        <li><strong>LSTM:</strong> Long Short-Term Memory, jenis RNN yang mampu menangani data urutan dengan dependensi jangka panjang.</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)

elif menu == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Halaman ini digunakan untuk melakukan analisis eksplorasi data.")
    
    # Load dataset
    data = pd.read_csv('/Users/muhammadzuamaalamin/Documents/labbelajar2new/project/sentimen/tweet.csv')
    
    # Display the first few rows of the dataset
    st.subheader("Data Awal")
    st.dataframe(data.head())
    
    # Display basic statistics
    st.subheader("Statistik Deskriptif")
    st.write(data.describe())
    
    # Display data types and null values
    st.subheader("Informasi Data")
    st.write(data.info())
    
    # Check for missing values
    st.subheader("Cek Nilai Kosong")
    st.write(data.isnull().sum())

    # Hapus Kolom yang tidak diperlukan
    data = data.drop(columns=['Unnamed: 0'])
    st.write("Setelah menghapus kolom yang tidak diperlukan:")
    st.dataframe(data.head())

    # Check for duplicate values
    st.subheader("Cek Nilai Duplikat")
    duplicate_count = data.duplicated().sum()
    st.write(f"Jumlah nilai duplikat: {duplicate_count}")
    
    if duplicate_count > 0:
        st.write("Contoh data duplikat:")
        st.dataframe(data[data.duplicated()])
    
    # Check for data balance
    st.subheader("Cek Keseimbangan Data")
    categories = ['Positif', 'Netral', 'Negatif']
    
    # Menghitung jumlah tweet untuk setiap kategori
    positif_count = len(data[data['sentimen'] == 'positif'])
    netral_count = len(data[data['sentimen'] == 'netral'])
    negatif_count = len(data[data['sentimen'] == 'negatif'])
    values = [positif_count, netral_count, negatif_count]
    
    # Membuat diagram batang menggunakan Matplotlib
    fig, ax = plt.subplots()
    ax.bar(categories, values, color=['green', 'blue', 'red'])
    
    # Menambahkan judul dan label
    ax.set_title('Distribusi Kategori Sentimen')
    ax.set_xlabel('Kategori')
    ax.set_ylabel('Jumlah Tweet')
    
    
    # Menampilkan diagram batang di Streamlit
    st.pyplot(fig)
    st.write("Jumlah tweet untuk setiap kategori:")
    st.write(f"Positif: {positif_count}, Netral: {netral_count}, Negatif: {negatif_count}")


elif menu == "Prepocessing dan Training":
    st.title("Preprocessing dan Training Model")
    st.write("Halaman ini digunakan untuk melakukan preprocessing data dan melatih model.")
    
    # Load dataset
    data = pd.read_csv('/Users/muhammadzuamaalamin/Documents/labbelajar2new/project/sentimen/tweet.csv')
    
    # Preprocessing
    st.subheader("Preprocessing Data")
    data["tweet"] = data["tweet"].apply(preprocessing)
    st.write("Contoh teks setelah preprocessing:")
    st.dataframe(data[['tweet']].head())
    
    # Split data into training and testing sets
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['sentimen'])
    X = data['tweet'].values
    
    # Vectorization
    tv = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), use_idf=True)
    X = tv.fit_transform(data['tweet']).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    
    # Membuat tabel perbandingan hasil train dan evaluasi
    st.subheader("Perbandingan Hasil Train dan Evaluasi Model")
    
    # Data hasil evaluasi
    evaluation_data = {
        "Model": ["Naive Bayes", "SVM", "LSTM", "GRU"],
        "Accuracy": [0.6419, 0.6143, 0.5702, 0.5675],
        "Precision": [0.6509, 0.6178, 0.5724, 0.5813],
        "Recall": [0.6457, 0.6143, 0.5702, 0.5675],
        "F1 Score": [0.6417, 0.6117, 0.5710, 0.5469]
    }
    
    # Konversi ke DataFrame
    evaluation_df = pd.DataFrame(evaluation_data)
    
    # Tampilkan tabel
    st.dataframe(evaluation_df)
    
    # Visualisasi hasil evaluasi
    st.subheader("Visualisasi Perbandingan Model")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    evaluation_df.set_index("Model").plot(kind="bar", ax=ax)
    
    # Menambahkan judul dan label
    ax.set_title("Perbandingan Hasil Evaluasi Model", fontsize=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Skor", fontsize=12)
    ax.legend(title="Metrik", fontsize=10)
    
    # Menampilkan visualisasi di Streamlit
    st.pyplot(fig)


elif menu == "Prediksi Sentimen":
    st.title("Prediksi Sentimen")
    st.write("Halaman ini digunakan untuk melakukan prediksi sentimen berdasarkan input manual.")

    # Load the trained models and TfidfVectorizer
    naive_bayes_model = joblib.load('./model/naive_bayes_model.pkl')
    svm_model = joblib.load('./model/svm_model_basic.pkl')
    vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')
    label_encoder = joblib.load('./model/label_encoder.pkl')
    lstm_model = load_model('./model/sentiment_analysis_lstm_model.h5')
    gru_model = load_model('./model/sentiment_analysis_gru_model.h5')

    # Input manual dengan satu teks
    manual_input = st.text_input("Masukkan teks untuk prediksi sentimen:")

    # Pilih jenis model yang akan digunakan
    model_type = st.radio("Pilih jenis model:", ("Konvensional", "Deep Learning"))

    if model_type == "Konvensional":
        # Pilih model konvensional
        model_choice = st.radio("Pilih model konvensional untuk prediksi:", ("Naive Bayes", "SVM"))

        if st.button("Prediksi"):
            if manual_input:
                # Preprocessing input manual (gunakan fungsi preprocessing yang sudah ada)
                processed_input = preprocessing(manual_input)

                # Transform input menggunakan TfidfVectorizer yang sudah dilatih
                transformed_input = vectorizer.transform([processed_input])

                # Prediksi menggunakan model yang dipilih
                if model_choice == "Naive Bayes":
                    predicted_label = naive_bayes_model.predict(transformed_input)
                elif model_choice == "SVM":
                    predicted_label = svm_model.predict(transformed_input)

                # Konversi label numerik ke label asli menggunakan LabelEncoder
                predicted_sentiment = label_encoder.inverse_transform(predicted_label)

                # Menampilkan hasil
                st.subheader("Hasil Prediksi")
                st.write(f"Tweet: {manual_input}")
                st.write(f"Model yang digunakan: {model_choice}")
                st.write(f"Prediksi Sentimen: {predicted_sentiment[0]}")

    elif model_type == "Deep Learning":

        tokenizer = joblib.load('./model/tokenizer.pkl')
        
        # Parameter preprocessing
        max_length = 30
        trunc_type = 'post'
        categories = ['Positif', 'Netral', 'Negatif']

        # Pilih model deep learning
        model_choice = st.radio("Pilih model deep learning untuk prediksi:", ("LSTM", "GRU"))

        # Load model yang sesuai
        if model_choice == "LSTM":
            model = lstm_model
        else:
            model = gru_model
        # Fungsi prediksi
        def predict_sentiment(input_text):
            # Preprocessing teks
            processed_text = preprocessing(input_text)

            # Konversi ke sequence
            sequence = tokenizer.texts_to_sequences([processed_text])
            
            # Cek apakah sequence valid
            if not sequence or not sequence[0]:
                return "Teks tidak dikenali oleh tokenizer."

            # Padding
            padded_sequence = pad_sequences(sequence, maxlen=max_length, truncating=trunc_type)

            # Prediksi
            prediction = model.predict(padded_sequence)
            predicted_class = prediction.argmax(axis=1)[0]

            return categories[predicted_class]

        # Tombol prediksi ditekan
        if st.button("Prediksi"):
            if manual_input:
                # Prediksi sentimen
                predicted_sentiment = predict_sentiment(manual_input)

                # Tampilkan hasil
                st.subheader("Hasil Prediksi")
                st.write(f"Tweet: {manual_input}")
                st.write(f"Model yang digunakan: {model_choice}")
                st.write(f"Prediksi Sentimen: {predicted_sentiment}")
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")

