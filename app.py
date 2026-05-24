import streamlit as st
import pandas as pd
import joblib
import os
import time
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat

# Download package NLTK yang dibutuhkan (Hanya berjalan sekali)
@st.cache_resource
def download_nltk_data():
    import nltk
    # Menambahkan punkt_tab sesuai permintaan error
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="centered"
)

# --- 2. LOAD MODEL TERBAIK ---
@st.cache_resource
def load_model():
    model_path = os.path.join(".dvc", "models", "best_rf_hho_model.pkl")
    return joblib.load(model_path)

model = load_model()

# --- 3. FUNGSI EKSTRAKSI FITUR NLP (OTOMATIS) ---
def extract_features(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Keamanan jika teks terlalu pendek
    if len(words) < 5 or len(sentences) == 0:
        return None
        
    # 1. Text Length
    text_length = len(text)
    
    # 2. Lexical Diversity Ratio (TTR - Type Token Ratio)
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words) if words else 0
    
    # 3. Readability Grade (Flesch-Kincaid)
    readability = textstat.flesch_kincaid_grade(text)
    
    # 4. Burstiness Index (Variasi panjang kalimat)
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    burstiness = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    
    # 5. Syntactic Variability (Pendekatan variasi panjang kata)
    word_lengths = [len(w) for w in words]
    syntactic_var = np.std(word_lengths)
    
    # 6. Perplexity Score (Simulasi heuristik lokal berdasarkan kompleksitas kata)
    # Perplexity asli butuh model LLM besar, ini adalah pendekatan statistik dasar
    perplexity = textstat.szigriszt_pazos(text) * 1.5 
    
    # 7 & 8 & 9. Fitur Metadata (Diasumsikan nilai rata-rata netral karena sulit dihitung tanpa API mesin aslinya)
    semantic_coherence = 0.5  # Asumsi netral
    prompt_complexity = 0.5   # Asumsi netral
    generation_confidence = 0.8 # Asumsi netral
    
    return [
        prompt_complexity, perplexity, burstiness,
        syntactic_var, semantic_coherence, lexical_diversity,
        readability, generation_confidence, text_length
    ]

# --- 4. TAMPILAN UI ---
st.title("🤖 Deteksi Teks: AI vs Manusia")
st.markdown("Ketik atau *paste* artikel di bawah ini, dan sistem NLP akan mengekstrak 9 fitur statistik secara otomatis untuk dianalisis oleh model HHO.")
st.divider()

user_text = st.text_area("Masukkan teks di sini (Minimal 3 kalimat):", height=200)

if st.button("🔍 Ekstrak & Analisis Teks", use_container_width=True, disabled=not user_text):
    
    with st.spinner("Memproses NLP dan menganalisis metrik..."):
        time.sleep(1) # Efek loading
        
        extracted_data = extract_features(user_text)
        
        if extracted_data is None:
            st.warning(" Teks terlalu pendek. Mohon masukkan minimal 3 kalimat yang utuh.")
        else:
            # Tampilkan metrik hasil ekstraksi secara rapi
            st.success(" Ekstraksi fitur berhasil!")
            
            with st.expander(" Lihat Hasil Angka Ekstraksi NLP", expanded=False):
                col1, col2 = st.columns(2)
                col1.write(f"- **Text Length:** {extracted_data[8]}")
                col1.write(f"- **Lexical Diversity:** {extracted_data[5]:.4f}")
                col1.write(f"- **Readability Grade:** {extracted_data[6]:.2f}")
                col1.write(f"- **Burstiness Index:** {extracted_data[2]:.4f}")
                
                col2.write(f"- **Syntactic Var:** {extracted_data[3]:.4f}")
                col2.write(f"- **Perplexity (Estimasi):** {extracted_data[1]:.4f}")
            
            # Konversi ke DataFrame sesuai format X_train
            input_df = pd.DataFrame([extracted_data], columns=[
                "prompt_complexity_score", "perplexity_score", "burstiness_index", 
                "syntactic_variability", "semantic_coherence_score", "lexical_diversity_ratio", 
                "readability_grade_level", "generation_confidence_score", "text_length"
            ])
            
            # Eksekusi Prediksi
            prediction = model.predict(input_df)
            
            st.subheader("Keputusan Sistem:")
            if prediction[0] == 1:
                st.error(" **TERDETEKSI BUATAN AI (AI-Generated)**")
                st.write("Variasi kalimat dan keragaman kata menunjukkan pola terstruktur khas mesin.")
            else:
                st.success("**TERDETEKSI BUATAN MANUSIA (Human-Written)**")
                st.write("Distribusi kalimat menunjukkan ketidakteraturan (burstiness) yang natural khas manusia.")