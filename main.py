import streamlit as st

# --- KONFIGURASI HALAMAN UTAMA ---
st.set_page_config(page_title="Smart Studio Arsitek", layout="wide", page_icon="🏗️")

# --- NAVIGASI ANTAR APLIKASI ---
# Ini fitur baru Streamlit untuk memanggil file lain sebagai halaman
pg = st.navigation([
    st.Page("app_pclp.py", title="Desain Sipil (PCLP)", icon="🚜"),
    st.Page("app_pclp_hidro.py", title="Analisis Hidrologi", icon="💧"),
])

st.sidebar.title("Navigasi Studio")
st.sidebar.info("Pilih modul aplikasi di atas.")

# --- JALANKAN NAVIGASI ---
pg.run()