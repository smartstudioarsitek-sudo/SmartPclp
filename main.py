import streamlit as st

# --- NAVIGASI UTAMA ---
# main.py bertugas sebagai "Launcher" atau Pintu Masuk
st.set_page_config(page_title="Smart Studio Arsitek", layout="wide", page_icon="🏗️")

pg = st.navigation([
    st.Page("app_pclp.py", title="Desain Sipil (PCLP)", icon="🚜"),
    st.Page("app_pclp_hidro.py", title="Analisis Hidrologi", icon="💧"),
])

st.sidebar.title("Navigasi Studio")
pg.run()
