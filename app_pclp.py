import streamlit as st
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. PARSER KHUSUS (Penerjemah Data Horizontal)
# ==========================================
def parse_pclp_horizontal(df):
    """
    Membaca format PCLP lama yang datanya melebar ke samping (Row X & Row Y).
    """
    parsed_data = []
    i = 0
    while i < len(df):
        row = df.iloc[i]
        
        # Cari marker 'X' di kolom index 3 (Kolom D)
        # Kita pakai try-except agar tidak error jika kolom kurang
        try:
            val_x = str(row[3]).strip().upper() if len(row) > 3 else ""
        except:
            val_x = ""

        if val_x == 'X':
            x_row = row
            
            # Cek baris bawahnya untuk 'Y'
            if i + 1 < len(df):
                y_row = df.iloc[i+1]
                try:
                    val_y = str(y_row[3]).strip().upper() if len(y_row) > 3 else ""
                except:
                    val_y = ""

                if val_y == 'Y':
                    # --- KETEMU PASANGAN X & Y ---
                    
                    # 1. Ambil Nama STA (Biasanya di baris Y, kolom index 1 / Kolom B)
                    raw_sta = str(y_row[1]).strip()
                    
                    # Bersihkan nama: hapus .0 di belakang angka (misal 1.0 jadi 1)
                    if raw_sta.endswith('.0'):
                        raw_sta = raw_sta[:-2]
                    
                    sta_name = raw_sta if raw_sta.lower() != 'nan' else f"Unknown_{i}"
                    
                    # 2. Ambil Data Angka (Mulai dari kolom index 4 / Kolom E)
                    # Pastikan kita tidak mengambil index di luar batas
                    max_len = min(len(x_row), len(y_row))
                    x_vals = x_row[4:max_len].values
                    y_vals = y_row[4:max_len].values
                    
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
                            'points': points,
                            'original_index': i
                        })
                    
                    i += 1 # Skip baris Y
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
    
    doc.layers.add(name='TANAH_ASLI', color=8)       
    doc.layers.add(name='DESAIN_SALURAN', color=1)   
    doc.layers.add(name='TEKS_DATA', color=7)        

    count = 0
    jarak_antar_gambar = 50 
    
    for item in all_results:
        sta = item['STA']
        tanah = item['points_tanah']
        desain = item['points_desain']
        
        offset_vec = (count * jarak_antar_gambar, 0)
        tanah_draw = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in tanah]
        desain_draw = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in desain]

        msp.add_lwpolyline(tanah_draw, dxfattribs={'layer': 'TANAH_ASLI'})
        msp.add_lwpolyline(desain_draw, dxfattribs={'layer': 'DESAIN_SALURAN'})

        info_txt = f"STA: {sta}\nCut: {item['cut']:.2f} m2\nFill: {item['fill']:.2f} m2"
        center_x = sum(p[0] for p in tanah_draw) / len(tanah_draw)
        max_y = max(p[1] for p in tanah_draw)
        
        msp.add_mtext(info_txt, dxfattribs={'char_height': 0.4, 'layer': 'TEKS_DATA'}).set_location(
            insert=(center_x, max_y + 2.0),
            attachment_point=ezdxf.const.MTEXT_TOP_CENTER
        )
        count += 1

    output = io.StringIO()
    doc.write(output)
    return output.getvalue().encode('utf-8')

# ==========================================
# 3. INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="PCLP Modern Auto", layout="wide")
st.title("🚜 PCLP Modern: Smart Match")

st.sidebar.header("1. Upload File")
uploaded_file = st.sidebar.file_uploader("Upload Excel PCLP (.xls / .xlsx)", type=["xls", "xlsx"])

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        st.sidebar.success("File OK!")
        
        # Konfigurasi Sheet
        st.sidebar.subheader("2. Pilih Sheet")
        sheet_ogl = st.sidebar.selectbox("Sheet Tanah (OGL):", sheet_names, index=0)
        
        # Coba auto-select sheet desain (biasanya ada kata 'Design' atau 'Plan')
        idx_des = 1 if len(sheet_names) > 1 else 0
        for i, name in enumerate(sheet_names):
            if "design" in name.lower(): idx_des = i
            
        sheet_desain = st.sidebar.selectbox("Sheet Desain:", sheet_names, index=idx_des)
        
        # KONFIGURASI MATCHING
        st.sidebar.subheader("3. Metode Pencocokan")
        match_mode = st.sidebar.radio(
            "Cara menjodohkan STA:",
            ("Otomatis (Berdasarkan Nama)", "Paksa Urutan (Abaikan Nama)")
        )
        st.sidebar.caption("Pilih 'Paksa Urutan' jika nama STA beda format (misal P0 vs 0+000).")

        if st.sidebar.button("🚀 PROSES DATA"):
            with st.spinner("Menganalisis data..."):
                
                # Baca Data
                df_ogl_raw = pd.read_excel(uploaded_file, sheet_name=sheet_ogl, header=None)
                df_des_raw = pd.read_excel(uploaded_file, sheet_name=sheet_desain, header=None)
                
                # Parsing
                data_ogl = parse_pclp_horizontal(df_ogl_raw)
                data_desain = parse_pclp_horizontal(df_des_raw)
                
                # Debug Info
                with st.expander("🕵️ Lihat Hasil Deteksi STA (Klik untuk buka)"):
                    c1, c2 = st.columns(2)
                    c1.write(f"**Tanah ({len(data_ogl)} data):**")
                    c1.write([d['STA'] for d in data_ogl])
                    c2.write(f"**Desain ({len(data_desain)} data):**")
                    c2.write([d['STA'] for d in data_desain])

                if not data_ogl:
                    st.error("Gagal membaca data Tanah! Pastikan ada baris dengan huruf 'X' dan 'Y' di kolom D.")
                    st.stop()

                # LOGIKA MATCHING
                results = []
                
                if match_mode == "Paksa Urutan (Abaikan Nama)":
                    # Mode: Pasangkan 1 lawan 1
                    jumlah = min(len(data_ogl), len(data_desain))
                    st.info(f"Mode Paksa Aktif: Memproses {jumlah} pasangan data pertama.")
                    
                    for i in range(jumlah):
                        item_t = data_ogl[i]
                        item_d = data_desain[i]
                        
                        cut, fill = hitung_cut_fill(item_t['points'], item_d['points'])
                        results.append({
                            'STA': item_t['STA'], # Pakai nama dari Tanah
                            'cut': cut,
                            'fill': fill,
                            'points_tanah': item_t['points'],
                            'points_desain': item_d['points']
                        })
                else:
                    # Mode: Nama Harus Sama
                    for item_t in data_ogl:
                        sta_t = item_t['STA'].strip().lower()
                        # Cari di desain yang namanya mirip
                        item_d = next((d for d in data_desain if d['STA'].strip().lower() == sta_t), None)
                        
                        if item_d:
                            cut, fill = hitung_cut_fill(item_t['points'], item_d['points'])
                            results.append({
                                'STA': item_t['STA'],
                                'cut': cut,
                                'fill': fill,
                                'points_tanah': item_t['points'],
                                'points_desain': item_d['points']
                            })
                
                # HASIL AKHIR
                if not results:
                    st.error("Tidak ditemukan pasangan STA! Coba ganti metode ke 'Paksa Urutan'.")
                else:
                    st.success(f"Sukses! {len(results)} Cross Section terhitung.")
                    
                    tab1, tab2 = st.tabs(["📊 Tabel Volume", "🖼️ Visual & Download"])
                    
                    with tab1:
                        df_res = pd.DataFrame(results)[['STA', 'cut', 'fill']]
                        df_res.columns = ['STA', 'Cut (m2)', 'Fill (m2)']
                        st.dataframe(df_res, use_container_width=True)
                        
                        tot_c = df_res['Cut (m2)'].sum()
                        tot_f = df_res['Fill (m2)'].sum()
                        st.metric("Total Volume (Asumsi Jarak 1m)", f"Cut: {tot_c:.2f} | Fill: {tot_f:.2f}")

                    with tab2:
                        if results:
                            # Preview Data Pertama
                            item = results[0]
                            fig, ax = plt.subplots(figsize=(10, 3))
                            tx, ty = zip(*item['points_tanah'])
                            dx, dy = zip(*item['points_desain'])
                            ax.plot(tx, ty, 'k--', label='Tanah')
                            ax.fill_between(tx, ty, min(ty)-2, color='gray', alpha=0.1)
                            ax.plot(dx, dy, 'r-', label='Desain')
                            ax.set_title(f"Preview: {item['STA']}")
                            ax.legend()
                            st.pyplot(fig)
                            
                            # Download
                            dxf_data = generate_dxf_batch(results)
                            st.download_button("📥 DOWNLOAD DXF (AutoCAD)", dxf_data, "Hasil_PCLP.dxf", "application/dxf")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
