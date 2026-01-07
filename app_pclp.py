import streamlit as st
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. ENGINE PERHITUNGAN & GAMBAR (BACKEND)
# ==========================================

def hitung_single_sta(tanah_df, desain_df):
    """
    Menghitung Cut/Fill untuk satu STA tertentu.
    Data input berupa DataFrame pandas.
    """
    # Urutkan berdasarkan jarak (X) dari kiri ke kanan
    tanah_df = tanah_df.sort_values(by='X')
    desain_df = desain_df.sort_values(by='X')

    # Konversi ke List of Tuples [(x,y), ...]
    tanah_pts = list(zip(tanah_df['X'], tanah_df['Y']))
    desain_pts = list(zip(desain_df['X'], desain_df['Y']))

    if not tanah_pts or not desain_pts:
        return 0, 0, [], []

    # Buat Datum (Dasar) untuk menutup poligon
    all_y = [p[1] for p in tanah_pts] + [p[1] for p in desain_pts]
    datum = min(all_y) - 5.0

    # Buat Poligon Shapely
    poly_tanah = Polygon(tanah_pts + [(tanah_pts[-1][0], datum), (tanah_pts[0][0], datum)])
    poly_desain = Polygon(desain_pts + [(desain_pts[-1][0], datum), (desain_pts[0][0], datum)])

    # Validasi Geometri
    if not poly_tanah.is_valid: poly_tanah = poly_tanah.buffer(0)
    if not poly_desain.is_valid: poly_desain = poly_desain.buffer(0)

    try:
        area_cut = poly_desain.intersection(poly_tanah).area
        area_fill = poly_desain.difference(poly_tanah).area
    except:
        area_cut, area_fill = 0, 0
    
    return area_cut, area_fill, tanah_pts, desain_pts

def generate_dxf_batch(all_results):
    """
    Membuat satu file DXF berisi BANYAK cross section yang disusun rapi.
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Setup Layers
    doc.layers.add(name='TANAH_ASLI', color=8)       # Abu-abu
    doc.layers.add(name='DESAIN_SALURAN', color=1)   # Merah
    doc.layers.add(name='TEKS_DATA', color=7)        # Putih

    # Konfigurasi Layout Gambar
    x_origin = 0
    y_origin = 0
    jarak_antar_gambar_x = 30  # Jarak horizontal antar gambar
    jarak_baris_y = 30         # Jika mau multiline (opsional, saat ini linear)
    
    count = 0
    
    for item in all_results:
        sta = item['STA']
        tanah = item['points_tanah']
        desain = item['points_desain']
        cut = item['cut']
        fill = item['fill']

        # Geser koordinat (Offsetting) agar gambar tidak menumpuk
        # Kita geser ke Kanan (X) setiap ganti STA
        offset_vec = (count * jarak_antar_gambar_x, 0)
        
        # Transformasi titik
        tanah_draw = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in tanah]
        desain_draw = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in desain]

        # Gambar Garis
        msp.add_lwpolyline(tanah_draw, dxfattribs={'layer': 'TANAH_ASLI'})
        msp.add_lwpolyline(desain_draw, dxfattribs={'layer': 'DESAIN_SALURAN'})

        # Gambar Teks Info
        info_txt = f"STA: {sta}\nCut: {cut:.2f} m2\nFill: {fill:.2f} m2"
        
        # Cari posisi teks (di atas tengah gambar)
        if tanah_draw:
            center_x = sum(p[0] for p in tanah_draw) / len(tanah_draw)
            max_y = max(p[1] for p in tanah_draw)
            
            msp.add_mtext(info_txt, dxfattribs={'char_height': 0.4, 'layer': 'TEKS_DATA'}).set_location(
                insert=(center_x, max_y + 2.0),
                attachment_point=ezdxf.const.MTEXT_TOP_CENTER
            )

        count += 1

    # Output ke Memory buffer
    output = io.StringIO()
    doc.write(output)
    return output.getvalue().encode('utf-8')

# ==========================================
# 2. TAMPILAN APLIKASI (FRONTEND)
# ==========================================

st.set_page_config(page_title="PCLP Pro", layout="wide")
st.title("🚜 PCLP Pro: Batch Cut & Fill Processor")
st.markdown("---")

# --- SIDEBAR: KONFIGURASI INPUT ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload File Excel (.xls / .xlsx)", type=["xls", "xlsx"])

if uploaded_file:
    try:
        # Baca semua sheet dalam Excel
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        st.sidebar.success("File berhasil dibaca!")
        st.sidebar.markdown("---")
        
        # --- MAPPING SHEET TANAH (OGL) ---
        st.sidebar.subheader("Konfigurasi Data Tanah (OGL)")
        sheet_ogl = st.sidebar.selectbox("Pilih Sheet Tanah Asli:", sheet_names, index=0)
        
        # Baca sheet terpilih untuk ambil nama kolom
        df_ogl_preview = pd.read_excel(uploaded_file, sheet_name=sheet_ogl, nrows=5)
        cols_ogl = df_ogl_preview.columns.tolist()
        
        col_sta_ogl = st.sidebar.selectbox("Kolom STA (Tanah):", cols_ogl)
        col_dist_ogl = st.sidebar.selectbox("Kolom Jarak/Offset (Tanah):", cols_ogl)
        col_elev_ogl = st.sidebar.selectbox("Kolom Elevasi (Tanah):", cols_ogl)
        
        st.sidebar.markdown("---")

        # --- MAPPING SHEET DESAIN ---
        st.sidebar.subheader("Konfigurasi Data Desain")
        sheet_desain = st.sidebar.selectbox("Pilih Sheet Desain:", sheet_names, index=min(1, len(sheet_names)-1))
        
        df_des_preview = pd.read_excel(uploaded_file, sheet_name=sheet_desain, nrows=5)
        cols_des = df_des_preview.columns.tolist()
        
        col_sta_des = st.sidebar.selectbox("Kolom STA (Desain):", cols_des)
        col_dist_des = st.sidebar.selectbox("Kolom Jarak/Offset (Desain):", cols_des)
        col_elev_des = st.sidebar.selectbox("Kolom Elevasi (Desain):", cols_des)
        
        st.sidebar.markdown("---")
        btn_process = st.sidebar.button("🚀 MULAI PROSES")

        if btn_process:
            # --- MULAI PROSES BATCH ---
            with st.spinner("Sedang memproses seluruh data STA..."):
                # 1. Load Full Data
                df_ogl = pd.read_excel(uploaded_file, sheet_name=sheet_ogl)
                df_des = pd.read_excel(uploaded_file, sheet_name=sheet_desain)
                
                # 2. Rename Kolom agar standar
                df_ogl = df_ogl.rename(columns={col_sta_ogl: 'STA', col_dist_ogl: 'X', col_elev_ogl: 'Y'})
                df_des = df_des.rename(columns={col_sta_des: 'STA', col_dist_des: 'X', col_elev_des: 'Y'})
                
                # 3. Pastikan format angka
                for df in [df_ogl, df_des]:
                    df['X'] = pd.to_numeric(df['X'], errors='coerce')
                    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
                
                # 4. Ambil List STA Unik (Intersection dari kedua data)
                sta_list = sorted(list(set(df_ogl['STA'].unique()) & set(df_des['STA'].unique())))
                
                if not sta_list:
                    st.error("Tidak ditemukan STA yang cocok antara Data Tanah dan Desain! Cek penulisan STA.")
                else:
                    # 5. Looping Perhitungan
                    results = []
                    summary_data = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, sta in enumerate(sta_list):
                        # Filter data per STA
                        sub_ogl = df_ogl[df_ogl['STA'] == sta]
                        sub_des = df_des[df_des['STA'] == sta]
                        
                        # Hitung
                        c, f, pts_t, pts_d = hitung_single_sta(sub_ogl, sub_des)
                        
                        # Simpan Hasil
                        results.append({
                            'STA': sta,
                            'cut': c,
                            'fill': f,
                            'points_tanah': pts_t,
                            'points_desain': pts_d
                        })
                        
                        summary_data.append({'STA': sta, 'Cut (m2)': c, 'Fill (m2)': f})
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(sta_list))
                    
                    # --- OUTPUT RESULT ---
                    st.success(f"Selesai! Berhasil memproses {len(sta_list)} Cross Section.")
                    
                    # Tabulasi
                    tab1, tab2 = st.tabs(["📊 Tabel Volume", "🖼️ Preview & Download"])
                    
                    with tab1:
                        df_res = pd.DataFrame(summary_data)
                        st.dataframe(df_res, use_container_width=True)
                        
                        total_cut = df_res['Cut (m2)'].sum()
                        total_fill = df_res['Fill (m2)'].sum()
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Total Volume Galian", f"{total_cut:,.2f} m3")
                        c2.metric("Total Volume Timbunan", f"{total_fill:,.2f} m3")
                        st.caption("*Asumsi jarak antar patok 1 meter (Volume = Luas x 1). Untuk volume real, kalikan dengan jarak antar patok.*")

                    with tab2:
                        # Dropdown untuk pilih preview
                        selected_sta = st.selectbox("Pilih STA untuk Preview:", sta_list)
                        
                        # Cari data STA terpilih
                        res_sta = next(item for item in results if item["STA"] == selected_sta)
                        
                        # Plot Matplotlib
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        t_x, t_y = zip(*res_sta['points_tanah'])
                        d_x, d_y = zip(*res_sta['points_desain'])
                        
                        ax.plot(t_x, t_y, label='Tanah Asli', color='gray', linestyle='--')
                        ax.fill_between(t_x, t_y, min(t_y)-5, color='gray', alpha=0.1)
                        ax.plot(d_x, d_y, label='Desain', color='red', linewidth=2)
                        
                        ax.set_title(f"Cross Section STA {selected_sta} (Cut: {res_sta['cut']:.2f} | Fill: {res_sta['fill']:.2f})")
                        ax.legend()
                        ax.grid(True, linestyle=':', alpha=0.5)
                        ax.set_aspect('equal')
                        
                        st.pyplot(fig)
                        
                        st.markdown("### Download Hasil")
                        # Generate DXF
                        dxf_bytes = generate_dxf_batch(results)
                        
                        st.download_button(
                            label="📥 Download Semua Gambar (.dxf)",
                            data=dxf_bytes,
                            file_name="All_Cross_Sections.dxf",
                            mime="application/dxf"
                        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        st.info("Pastikan file Excel memiliki header kolom di baris pertama.")

else:
    st.info("👈 Silakan upload file Excel PCLP (misal: Cross_ogl.xls) di menu sebelah kiri.")
    st.markdown("""
    **Format Data yang Didukung:**
    Satu file Excel yang berisi minimal 2 Sheet:
    1. **Sheet Tanah:** Berisi kolom STA, Jarak, Elevasi Tanah.
    2. **Sheet Desain:** Berisi kolom STA, Jarak, Elevasi Desain.
    """)
