import streamlit as st
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. PARSER CERDAS (AUTO SCAN X & Y)
# ==========================================
def parse_pclp_smart(df):
    """
    Mencari pasangan baris X dan Y di kolom mana saja.
    Tidak peduli X ada di kolom A, B, C, atau D, akan otomatis ketemu.
    """
    parsed_data = []
    i = 0
    
    # Loop baris demi baris
    while i < len(df):
        row = df.iloc[i]
        
        # 1. SCANNING: Cari kolom mana yang isinya huruf 'X'
        x_col_idx = -1
        # Kita cek 10 kolom pertama saja (biasanya header ada di kiri)
        max_col_check = min(10, len(row))
        
        for c in range(max_col_check):
            try:
                val = str(row[c]).strip().upper()
                if val == 'X':
                    x_col_idx = c
                    break
            except:
                continue
        
        # 2. MATCHING: Jika ketemu X, cek baris bawahnya apakah Y?
        if x_col_idx != -1 and (i + 1 < len(df)):
            row_next = df.iloc[i+1]
            try:
                val_next = str(row_next[x_col_idx]).strip().upper()
            except:
                val_next = ""
                
            if val_next == 'Y':
                # --- KETEMU PASANGAN EMAS (X & Y) ---
                
                # A. Ambil Nama STA
                # Biasanya nama STA ada di sebelah kiri 'Y' (misal Y di kolom D, STA di kolom B)
                # Kita coba cari mundur dari kolom Y
                sta_name = f"Unknown_{i}"
                if x_col_idx >= 2:
                    val_sta = str(row_next[x_col_idx - 2]).strip() # Coba 2 kolom ke kiri
                    if val_sta and val_sta.lower() != 'nan':
                        sta_name = val_sta
                
                # Bersihkan nama STA (.0 di belakang angka)
                if sta_name.endswith('.0'): sta_name = sta_name[:-2]
                
                # B. Ambil Data Koordinat
                # Data dimulai dari kolom SETELAH X (x_col_idx + 1)
                start_data = x_col_idx + 1
                
                # Pastikan panjang baris sama
                max_len = min(len(row), len(row_next))
                x_vals = row[start_data:max_len].values
                y_vals = row_next[start_data:max_len].values
                
                points = []
                for x, y in zip(x_vals, y_vals):
                    try:
                        xf = float(x)
                        yf = float(y)
                        if not (pd.isna(xf) or pd.isna(yf)):
                            points.append((xf, yf))
                    except:
                        continue
                
                if points:
                    parsed_data.append({
                        'STA': sta_name,
                        'points': points
                    })
                
                # Lompat 1 baris (karena baris Y sudah dipakai)
                i += 1
                
        i += 1
        
    return parsed_data

# ==========================================
# 2. ENGINE PERHITUNGAN
# ==========================================
def hitung_cut_fill(tanah_pts, desain_pts):
    if not tanah_pts or not desain_pts:
        return 0, 0
    
    # Datum
    all_y = [p[1] for p in tanah_pts] + [p[1] for p in desain_pts]
    datum = min(all_y) - 5.0

    # Poligon
    poly_tanah = Polygon(tanah_pts + [(tanah_pts[-1][0], datum), (tanah_pts[0][0], datum)])
    poly_desain = Polygon(desain_pts + [(desain_pts[-1][0], datum), (desain_pts[0][0], datum)])

    if not poly_tanah.is_valid: poly_tanah = poly_tanah.buffer(0)
    if not poly_desain.is_valid: poly_desain = poly_desain.buffer(0)

    try:
        area_cut = poly_desain.intersection(poly_tanah).area
        area_fill = poly_desain.difference(poly_tanah).area
    except:
        area_cut, area_fill = 0, 0
    
    return area_cut, area_fill

def generate_dxf_batch(all_results):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Layers
    if 'TANAH_ASLI' not in doc.layers: doc.layers.add(name='TANAH_ASLI', color=8)
    if 'DESAIN_SALURAN' not in doc.layers: doc.layers.add(name='DESAIN_SALURAN', color=1)
    if 'TEKS_DATA' not in doc.layers: doc.layers.add(name='TEKS_DATA', color=7)

    count = 0
    jarak_antar_gambar = 60 # Jarak antar gambar di AutoCAD
    
    for item in all_results:
        sta = item['STA']
        tanah = item['points_tanah']
        desain = item['points_desain']
        
        offset_vec = (count * jarak_antar_gambar, 0)
        tanah_draw = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in tanah]
        desain_draw = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in desain]

        msp.add_lwpolyline(tanah_draw, dxfattribs={'layer': 'TANAH_ASLI'})
        msp.add_lwpolyline(desain_draw, dxfattribs={'layer': 'DESAIN_SALURAN'})

        info_txt = f"STA: {sta}\\PCut: {item['cut']:.2f} m2\\PFill: {item['fill']:.2f} m2"
        
        center_x = sum(p[0] for p in tanah_draw) / len(tanah_draw)
        max_y = max(p[1] for p in tanah_draw)
        
        msp.add_mtext(info_txt, dxfattribs={'char_height': 0.5, 'layer': 'TEKS_DATA'}).set_location(
            insert=(center_x, max_y + 3.0),
            attachment_point=ezdxf.const.MTEXT_TOP_CENTER
        )
        count += 1

    output = io.StringIO()
    doc.write(output)
    return output.getvalue().encode('utf-8')

# ==========================================
# 3. INTERFACE APLIKASI
# ==========================================
st.set_page_config(page_title="PCLP Ultimate", layout="wide")
st.title("🚜 PCLP Ultimate: Auto-Scan Engine")

st.markdown("""
<style>
div.stButton > button:first-child {background-color: #0099ff; color: white; font-size: 20px;}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel PCLP (.xls / .xlsx)", type=["xls", "xlsx"])

if uploaded_file:
    try:
        # Load Excel Wrapper
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        st.sidebar.success("✅ File Terbaca!")
        
        # Pilih Sheet
        st.sidebar.subheader("2. Pilih Sheet Data")
        sheet_ogl = st.sidebar.selectbox("Sheet Tanah Asli:", sheet_names, index=0)
        
        # Auto-detect sheet desain
        idx_des = 1 if len(sheet_names) > 1 else 0
        for i, name in enumerate(sheet_names):
            if "design" in name.lower() or "rencana" in name.lower(): idx_des = i
        sheet_desain = st.sidebar.selectbox("Sheet Desain:", sheet_names, index=idx_des)

        # Opsi Matching
        st.sidebar.subheader("3. Opsi Proses")
        match_mode = st.sidebar.radio("Metode Pasangkan STA:", 
                                      ["Paksa Urutan (Disarankan)", "Cocokkan Nama STA"])
        
        st.sidebar.info("Tips: Gunakan 'Paksa Urutan' jika nama STA di tanah & desain sering beda ketik.")

        if st.sidebar.button("🚀 MULAI PROSES"):
            with st.spinner("🔍 Sedang scanning posisi X dan Y di Excel..."):
                
                # Baca Raw Data (Tanpa Header)
                df_ogl_raw = pd.read_excel(uploaded_file, sheet_name=sheet_ogl, header=None)
                df_des_raw = pd.read_excel(uploaded_file, sheet_name=sheet_desain, header=None)
                
                # Parsing Smart
                data_ogl = parse_pclp_smart(df_ogl_raw)
                data_desain = parse_pclp_smart(df_des_raw)
                
                if not data_ogl:
                    st.error(f"❌ Tidak ditemukan data Tanah di sheet '{sheet_ogl}'! Pastikan ada baris dengan kode 'X' dan 'Y'.")
                    st.stop()
                
                if not data_desain:
                    st.error(f"❌ Tidak ditemukan data Desain di sheet '{sheet_desain}'!")
                    st.stop()
                
                st.success(f"Ditemukan: {len(data_ogl)} Profil Tanah & {len(data_desain)} Profil Desain.")

                # Proses Matching
                results = []
                
                if match_mode == "Paksa Urutan (Disarankan)":
                    limit = min(len(data_ogl), len(data_desain))
                    for i in range(limit):
                        item_t = data_ogl[i]
                        item_d = data_desain[i]
                        c, f = hitung_cut_fill(item_t['points'], item_d['points'])
                        results.append({
                            'STA': item_t['STA'],
                            'cut': c, 'fill': f,
                            'points_tanah': item_t['points'],
                            'points_desain': item_d['points']
                        })
                else:
                    # Match by Name
                    for item_t in data_ogl:
                        t_name = item_t['STA'].lower().replace(" ", "")
                        item_d = next((d for d in data_desain if d['STA'].lower().replace(" ", "") == t_name), None)
                        if item_d:
                            c, f = hitung_cut_fill(item_t['points'], item_d['points'])
                            results.append({
                                'STA': item_t['STA'],
                                'cut': c, 'fill': f,
                                'points_tanah': item_t['points'],
                                'points_desain': item_d['points']
                            })

                if not results:
                    st.warning("⚠️ Tidak ada STA yang cocok! Coba ubah metode ke 'Paksa Urutan'.")
                else:
                    # TAMPILKAN HASIL
                    tab1, tab2 = st.tabs(["📊 Data & Volume", "🖼️ Visualisasi & DWG"])
                    
                    with tab1:
                        df_res = pd.DataFrame(results)[['STA', 'cut', 'fill']]
                        df_res.columns = ['STA', 'Galian (m2)', 'Timbunan (m2)']
                        st.dataframe(df_res, use_container_width=True)
                        st.metric("Total Galian", f"{df_res['Galian (m2)'].sum():.2f} m3")

                    with tab2:
                        # Plot salah satu sebagai contoh
                        item = results[0]
                        fig, ax = plt.subplots(figsize=(10, 3))
                        tx, ty = zip(*item['points_tanah'])
                        dx, dy = zip(*item['points_desain'])
                        ax.plot(tx, ty, 'k-', linewidth=1, label='Tanah Asli')
                        ax.fill_between(tx, ty, min(ty)-2, color='gray', alpha=0.2)
                        ax.plot(dx, dy, 'r-', linewidth=2, label='Desain')
                        ax.set_title(f"Preview: {item['STA']}")
                        ax.legend()
                        ax.set_aspect('equal')
                        st.pyplot(fig)
                        
                        # Download Button
                        dxf_data = generate_dxf_batch(results)
                        st.download_button(
                            "📥 DOWNLOAD FILE AUTOCAD (.DXF)", 
                            dxf_data, 
                            "Hasil_PCLP_Auto.dxf", 
                            "application/dxf"
                        )

    except Exception as e:
        st.error(f"Terjadi Error: {e}")
        st.info("Jika error 'xlrd', pastikan library xlrd sudah terinstall (pip install xlrd).")
