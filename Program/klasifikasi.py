import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Fungsi untuk memuat data dari file Excel
def load_data():
    # Upload file Excel
    uploaded_file = st.file_uploader("Upload File Deskripsi Pemesanan (XLSX)", type="xlsx")
    if uploaded_file is not None:
        # Membaca data dari file Excel
        data = pd.read_excel(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data[['Description', 'Is secure cargo']])
        return data
    return None

# Fungsi untuk melatih model
def train_model(data):
    # Pastikan data memiliki kolom yang sesuai
    if 'Description' not in data or 'Is secure cargo' not in data:
        st.error("Data harus memiliki kolom 'Description' dan 'Is secure cargo'")
        return None, None
    
    # Fitur (Description) dan label (Is secure cargo)
    X = data['Description']
    y = data['Is secure cargo']
    
    # Mengubah teks menjadi fitur menggunakan TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    
    # Membagi data untuk pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
    
    # Melatih model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label='Secure')  # Anggap 'Secure' adalah label positif
    precision = precision_score(y_test, y_pred, pos_label='Secure')
    f1 = f1_score(y_test, y_pred, pos_label='Secure')
    cm = confusion_matrix(y_test, y_pred)

    # Menampilkan hasil evaluasi model
    st.write(f"Akurasi: {accuracy * 100:.2f}%")
    st.write(f"Recall: {recall * 100:.2f}%")
    st.write(f"Precision: {precision * 100:.2f}%")
    st.write(f"F1-Score: {f1 * 100:.2f}%")

    # Menampilkan confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Secure', 'Secure'], yticklabels=['Not Secure', 'Secure'])
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    st.pyplot(fig)
    
    return model, vectorizer

# Fungsi untuk memprediksi keamanan kargo
def predict_security(model, vectorizer):
    st.subheader("Masukkan Deskripsi Pemesanan untuk Prediksi Keamanan Kargo")
    deskripsi = st.text_area("Description")
    
    if deskripsi:
        deskripsi_tfidf = vectorizer.transform([deskripsi])
        prediksi = model.predict(deskripsi_tfidf)
        
        # Menampilkan hasil prediksi dengan penyesuaian label
        if prediksi[0] == 'Secure':
            st.write("Prediksi Keamanan Kargo: Aman")
        else:
            st.write("Prediksi Keamanan Kargo: Tidak Aman")

# Fungsi untuk analisis pola risiko pada kargo tidak aman
def analyze_risk_patterns(data):
    # Filter data untuk kargo yang tidak aman
    not_secure_data = data[data['Is secure cargo'] == 'Not Secure']
    
    # Menggunakan CountVectorizer untuk menghitung kata-kata yang sering muncul dalam deskripsi kargo yang tidak aman
    vectorizer = CountVectorizer(stop_words='english', max_features=20)  # Ambil 20 kata terbanyak
    X_not_secure = vectorizer.fit_transform(not_secure_data['Description'])
    
    # Ambil kata-kata yang sering muncul
    word_freq = pd.DataFrame(X_not_secure.toarray(), columns=vectorizer.get_feature_names_out())
    word_sum = word_freq.sum(axis=0).sort_values(ascending=False)
    
    # Menampilkan kata-kata yang sering muncul sebagai pola risiko
    st.subheader("Pola Risiko: Kata-Kata yang Sering Muncul pada Kargo Tidak Aman")
    st.write(word_sum)

    # Menampilkan WordCloud dari kata-kata yang sering muncul
    st.subheader("Word Cloud untuk Pola Risiko")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_sum)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

# Antarmuka Streamlit
st.title("Prediksi Keamanan Kargo dan Analisis Pola Risiko")

# Memuat data dari file Excel
data = load_data()

if data is not None:
    # Menampilkan analisis frekuensi kargo aman dan tidak aman
    st.subheader("Frekuensi Keamanan Kargo")
    keamanan_frek = data['Is secure cargo'].value_counts()
    st.write(keamanan_frek)

    # Menambahkan analisis pola risiko pada kargo tidak aman
    if st.button("Analisis Pola Risiko untuk Kargo Tidak Aman"):
        analyze_risk_patterns(data)

    # Latih model untuk prediksi keamanan kargo
    if st.button("Latih Model Prediksi Keamanan Kargo"):
        model, vectorizer = train_model(data)
        if model:
            st.success("Model berhasil dilatih!")
            
            # Fitur prediksi keamanan kargo
            predict_security(model, vectorizer)
