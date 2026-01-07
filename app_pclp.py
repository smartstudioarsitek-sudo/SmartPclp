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
    # Pastikan data dibaca tanpa header agar indeks baris terjaga
    parsed_data = []
    
    # Konversi ke numpy agar iterasi lebih cepat & aman
    # Kita cari baris yang mengandung marker 'X' di kolom ke-4 (index 3)
    # Struktur: 
    # Row i   : ... ... ... 'X'  x1  x2  x3 ...
    # Row i+1 : ... STA ... 'Y'  y1  y2  y3 ...
    
    i = 0
    while i < len(df):
        row = df.iloc[i]
        
        # Cek apakah baris ini baris 'X'
        # Kita cek kolom index 3 (Kolom D di Excel)
        if len(row) > 3 and str(row[3]).strip().upper() == 'X':
            x_row = row
            
            # Cek baris depannya untuk 'Y'
            if i + 1 < len(df):
                y_row = df.iloc[i+1]
                if len(y_row) > 3 and str(y_row[3]).strip().upper() == 'Y':
                    
                    # --- KETEMU PASANGAN X & Y ---
                    
                    # 1. Ambil Nama STA (Biasanya di baris Y, kolom index 1 / Kolom B)
                    raw_sta = str(y_row[1]).strip()
                    # Bersihkan nama STA (misal "1.0" jadi "P1" atau biarkan string)
                    sta_name = raw_sta if raw_sta != 'nan' else f"STA_{i}"
                    
                    # 2. Ambil Data Angka (Mulai dari kolom index 4 / Kolom E sampai habis)
                    x_vals = x_row[4:].values
                    y_vals = y_row[4:].values
                    
                    points = []
                    # Pasangkan X dan Y
                    for x, y in zip(x_vals, y_vals):
                        try:
                            xf = float(x)
                            yf = float(y)
                            # Simpan hanya jika angka valid (bukan NaN/Kosong)
                            if not (pd.isna(xf) or pd.isna(yf)):
                                points.append((xf, yf))
                        except:
                            continue
                    
                    # Simpan hasil parsing jika ada poin
                    if points:
                        parsed_data.append({
                            'STA': sta_name,
                            'points': points
                        })
                    
                    # Lompat 1 baris karena baris Y sudah diproses
                    i += 1
        i += 1
        
    return parsed_data

# ==========================================
# 2. ENGINE PERHITUNGAN
# ==========================================
def hitung_cut_fill(tanah_pts, desain_pts):
    if not tanah_pts or not desain_pts:
        return 0, 0
        
    # Buat Datum (Dasar)
    all_y = [p[1] for p in tanah_pts] + [p[1] for p in desain_pts]
    datum = min(all_y) - 5.0

    # Buat Poligon
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
    
    # Layering
    doc.layers.add(name='TANAH_ASLI', color=8)       
    doc.layers.add(name='DESAIN_SALURAN', color=1)   
    doc.layers.add(name='TEKS_DATA', color=7)        

    count = 0
    jarak_antar_gambar = 40  # Geser ke kanan setiap gambar baru
    
    for item in all_results:
        sta = item['STA']
        tanah = item['points_tanah']
        desain = item['points_desain']
        
        # Geser koordinat agar berjejer
        offset_vec = (count * jarak_antar_gambar, 0)
        tanah_draw = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in tanah]
        desain_draw = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in desain]

        # Gambar
        msp.add_lwpolyline(tanah_draw, dxfattribs={'layer': 'TANAH_ASLI'})
        msp.add_lwpolyline(desain_draw, dxfattribs={'layer': 'DESAIN_SALURAN'})

        # Teks Info
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
st.title("🚜 PCLP Modern: Auto-Detect Format")

st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload Excel PCLP (.xls / .xlsx)", type=["xls", "xlsx"])

if uploaded_file:
    try:
        # Baca nama sheet
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        st.sidebar.success("File terbaca!")
        st.sidebar.info("Aplikasi otomatis mendeteksi format Horizontal (PCLP Lama).")
        
        # Pilih Sheet
        sheet_ogl = st.sidebar.selectbox("Pilih Sheet Tanah Asli:", sheet_names, index=0)
        sheet_desain = st.sidebar.selectbox("Pilih Sheet Desain:", sheet_names, index=min(1, len(sheet_names)-1))
        
        if st.sidebar.button("🚀 PROSES DATA"):
            with st.spinner("Sedang membaca format PCLP Horizontal..."):
                
                # BACA TANAH (Header=None agar baris tidak hilang)
                df_ogl_raw = pd.read_excel(uploaded_file, sheet_name=sheet_ogl, header=None)
                data_ogl = parse_pclp_horizontal(df_ogl_raw)
                
                # BACA DESAIN
                df_des_raw = pd.read_excel(uploaded_file, sheet_name=sheet_desain, header=None)
                data_desain = parse_pclp_horizontal(df_des_raw)
                
                st.write(f"✅ Ditemukan **{len(data_ogl)}** data Tanah dan **{len(data_desain)}** data Desain.")
                
                # MATCHING DATA (Cari STA yang sama)
                results = []
                summary = []
                
                # Loop berdasarkan data Tanah yang ketemu
                for item_t in data_ogl:
                    sta_name = item_t['STA']
                    
                    # Cari pasangan di data Desain
                    # (Logika fuzzy: cari yang namanya persis sama)
                    item_d = next((d for d in data_desain if d['STA'] == sta_name), None)
                    
                    if item_d:
                        cut, fill = hitung_cut_fill(item_t['points'], item_d['points'])
                        
                        results.append({
                            'STA': sta_name,
                            'cut': cut,
                            'fill': fill,
                            'points_tanah': item_t['points'],
                            'points_desain': item_d['points']
                        })
                        summary.append({'STA': sta_name, 'Cut': cut, 'Fill': fill})
                
                if not results:
                    st.error("Tidak ada STA yang cocok namanya antara sheet Tanah dan Desain. Cek penulisan nama STA di Excel.")
                else:
                    st.success(f"Berhasil menghitung {len(results)} Cross Section!")
                    
                    # TABS OUTPUT
                    tab1, tab2 = st.tabs(["📊 Tabel Volume", "🖼️ Preview & Download"])
                    
                    with tab1:
                        st.dataframe(pd.DataFrame(summary), use_container_width=True)
                    
                    with tab2:
                        # Preview Random
                        preview_item = results[0]
                        fig, ax = plt.subplots(figsize=(10, 3))
                        tx, ty = zip(*preview_item['points_tanah'])
                        dx, dy = zip(*preview_item['points_desain'])
                        ax.plot(tx, ty, 'k--', label='Tanah')
                        ax.plot(dx, dy, 'r-', label='Desain')
                        ax.set_title(f"Preview STA {preview_item['STA']}")
                        ax.legend()
                        ax.set_aspect('equal')
                        st.pyplot(fig)
                        
                        # Download DXF
                        dxf_data = generate_dxf_batch(results)
                        st.download_button("📥 Download File DXF", dxf_data, "Cross_Section_Result.dxf", "application/dxf")

    except Exception as e:
        st.error(f"Error System: {e}")
