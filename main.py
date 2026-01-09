import streamlit as st

# --- KONFIGURASI UTAMA ---
st.set_page_config(page_title="Smart Studio Arsitek", layout="wide", page_icon="🏗️")

# --- NAVIGASI ---
# Pastikan file app_pclp.py dan app_pclp_hidro.py ada di folder yang sama
pg = st.navigation([
    st.Page("app_pclp.py", title="Desain Sipil (PCLP)", icon="🚜"),
    st.Page("app_pclp_hidro.py", title="Analisis Hidrologi", icon="💧"),
])

st.sidebar.title("Navigasi Studio")
pg.run()
