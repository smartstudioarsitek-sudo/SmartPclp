import streamlit as st
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, LineString, Point
import matplotlib.pyplot as plt
import io
import numpy as np

# Coba import library spasial (Handle error jika belum diinstall)
try:
    import geopandas as gpd
    import rasterio
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False

# ==========================================
# 1. CORE ENGINE: PARSER & MATH
# ==========================================

def parse_pclp_smart(df):
    """Parser universal untuk membaca format Horizontal PCLP (Cross & Long)."""
    parsed_data = []
    i = 0
    while i < len(df):
        row = df.iloc[i]
        x_col_idx = -1
        max_col_check = min(15, len(row))
        
        # Scan posisi 'X'
        for c in range(max_col_check):
            try:
                if str(row[c]).strip().upper() == 'X':
                    x_col_idx = c
                    break
            except: continue
        
        # Match dengan 'Y'
        if x_col_idx != -1 and (i + 1 < len(df)):
            try:
                if str(df.iloc[i+1][x_col_idx]).strip().upper() == 'Y':
                    # Ambil Nama (STA)
                    sta_name = f"STA_{i}"
                    if x_col_idx >= 2:
                        val_sta = str(df.iloc[i+1][x_col_idx - 2]).strip()
                        if val_sta and val_sta.lower() != 'nan': sta_name = val_sta
                    if sta_name.endswith('.0'): sta_name = sta_name[:-2]

                    # Ambil Data
                    start_data = x_col_idx + 1
                    max_len = min(len(row), len(df.iloc[i+1]))
                    x_vals = row[start_data:max_len].values
                    y_vals = df.iloc[i+1][start_data:max_len].values
                    
                    points = []
                    for x, y in zip(x_vals, y_vals):
                        try:
                            xf, yf = float(x), float(y)
                            if not (pd.isna(xf) or pd.isna(yf)): points.append((xf, yf))
                        except: continue
                    
                    if points:
                        parsed_data.append({'STA': sta_name, 'points': points})
                    i += 1
            except: pass
        i += 1
    return parsed_data

def hitung_cut_fill(tanah_pts, desain_pts):
    """Menghitung luas area Cut & Fill per cross section."""
    if not tanah_pts or not desain_pts: return 0, 0
    all_y = [p[1] for p in tanah_pts] + [p[1] for p in desain_pts]
    datum = min(all_y) - 5.0
    poly_tanah = Polygon(tanah_pts + [(tanah_pts[-1][0], datum), (tanah_pts[0][0], datum)]).buffer(0)
    poly_desain = Polygon(desain_pts + [(desain_pts[-1][0], datum), (desain_pts[0][0], datum)]).buffer(0)
    try:
        area_cut = poly_desain.intersection(poly_tanah).area
        area_fill = poly_desain.difference(poly_tanah).area
    except: area_cut, area_fill = 0, 0
    return area_cut, area_fill

# ==========================================
# 2. DXF GENERATORS
# ==========================================

def generate_dxf_cross(all_results):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add(name='TANAH_ASLI', color=8)
    doc.layers.add(name='DESAIN_SALURAN', color=1)
    doc.layers.add(name='TEKS_DATA', color=7)

    count = 0
    jarak_antar_gambar = 60 
    for item in all_results:
        offset_vec = (count * jarak_antar_gambar, 0)
        tanah = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in item['points_tanah']]
        desain = [(p[0] + offset_vec[0], p[1] + offset_vec[1]) for p in item['points_desain']]

        msp.add_lwpolyline(tanah, dxfattribs={'layer': 'TANAH_ASLI'})
        msp.add_lwpolyline(desain, dxfattribs={'layer': 'DESAIN_SALURAN'})
        
        info = f"{item['STA']}\\PCut: {item['cut']:.2f} m2\\PFill: {item['fill']:.2f} m2"
        cx = sum(p[0] for p in tanah)/len(tanah)
        my = max(p[1] for p in tanah)
        msp.add_mtext(info, dxfattribs={'char_height':0.5, 'layer':'TEKS_DATA'}).set_location(insert=(cx, my+3), attachment_point=ezdxf.const.MTEXT_TOP_CENTER)
        count += 1
    
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

def generate_dxf_long(merged_tanah, merged_desain):
    """Generator khusus Long Section (Profil Memanjang)."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add(name='LONG_TANAH', color=8)   # Abu-abu
    doc.layers.add(name='LONG_DESAIN', color=1)  # Merah

    # Gambar Garis Panjang
    if merged_tanah:
        msp.add_lwpolyline(merged_tanah, dxfattribs={'layer': 'LONG_TANAH'})
    if merged_desain:
        msp.add_lwpolyline(merged_desain, dxfattribs={'layer': 'LONG_DESAIN'})
    
    # Grid Sederhana (Opsional)
    if merged_tanah:
        min_x = min(p[0] for p in merged_tanah)
        max_x = max(p[0] for p in merged_tanah)
        min_y = min(min(p[1] for p in merged_tanah), min(p[1] for p in merged_desain))
        msp.add_line((min_x, min_y-2), (max_x, min_y-2), dxfattribs={'color': 7})
        msp.add_text(f"Long Section Profile (L={max_x-min_x:.1f}m)", dxfattribs={'height': 1.0}).set_placement((min_x, min_y-5))

    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

# ==========================================
# 3. GLOBAL MAPPER ENGINE
# ==========================================
def process_spatial_data(dem_file, align_file, step_m):
    """Sampling elevasi dari DEM berdasarkan garis Alinyemen (GeoJSON)."""
    if not HAS_GEO_LIBS:
        return None, "Library Geospasial tidak terinstall."

    try:
        # 1. Baca Alignment (GeoJSON)
        gdf = gpd.read_file(align_file)
        # Pastikan CRS PROJECTED (Meter), bukan LatLong. Jika LatLong, estimasi ke UTM (Auto-reproject logic simple)
        if gdf.crs.is_geographic:
            # Estimasi ke Pseudo-Mercator untuk hitungan meter (Simple method)
            gdf = gdf.to_crs(epsg=3857) 
        
        line_geom = gdf.geometry.iloc[0] # Ambil garis pertama
        if not isinstance(line_geom, LineString):
            return None, "File GeoJSON harus berisi tipe LineString (Garis)!"

        # 2. Baca DEM
        with rasterio.open(dem_file) as src:
            # Cek overlap
            # Note: Idealnya reproyeksi raster, tapi berat. Kita sampling koordinat world.
            pass # Asumsi user sudah upload data yg overlay-nya benar (sama2 UTM atau sama2 LatLong)

            # 3. Generate Long Section Data
            distances = np.arange(0, line_geom.length, step_m)
            long_points = []
            
            for dist in distances:
                pt = line_geom.interpolate(dist)
                # Sampling Raster
                # Transform point to raster CRS if needed (skipped for simplicity, assuming matching CRS)
                try:
                    vals = list(src.sample([(pt.x, pt.y)]))
                    elev = vals[0][0]
                    if elev != src.nodata:
                        long_points.append((dist, float(elev)))
                except: pass
            
            return long_points, None

    except Exception as e:
        return None, str(e)

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.set_page_config(page_title="PCLP Studio", layout="wide")
st.title("🚜 PCLP Studio: Engineering Suite")

# --- CSS Custom ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; }
    .stTabs [aria-selected="true"] { background-color: #0099ff; color: white; }
</style>""", unsafe_allow_html=True)

# --- TABS UTAMA ---
tab_cross, tab_long, tab_gis = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🌍 GLOBAL MAPPER"])

# ==========================================
# TAB 1: CROSS SECTION (Existing)
# ==========================================
with tab_cross:
    st.header("Analisis Potongan Melintang (Cut & Fill)")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        f_cross = st.file_uploader("Upload Excel Cross (.xls/.xlsx)", key="u_cross")
        if f_cross:
            xls = pd.ExcelFile(f_cross)
            s_ogl = st.selectbox("Sheet Tanah:", xls.sheet_names, key="s_c_ogl")
            s_des = st.selectbox("Sheet Desain:", xls.sheet_names, index=min(1,len(xls.sheet_names)-1), key="s_c_des")
            mode_cross = st.radio("Metode:", ["Paksa Urutan", "Match Nama"], key="m_cross")
            
            if st.button("Proses Cross Section", key="b_cross"):
                df_o = pd.read_excel(f_cross, sheet_name=s_ogl, header=None)
                df_d = pd.read_excel(f_cross, sheet_name=s_des, header=None)
                d_o = parse_pclp_smart(df_o)
                d_d = parse_pclp_smart(df_d)
                
                res = []
                iter_limit = min(len(d_o), len(d_d)) if mode_cross == "Paksa Urutan" else len(d_o)
                
                if mode_cross == "Paksa Urutan":
                    for i in range(iter_limit):
                        c, f = hitung_cut_fill(d_o[i]['points'], d_d[i]['points'])
                        res.append({'STA': d_o[i]['STA'], 'cut': c, 'fill': f, 'points_tanah': d_o[i]['points'], 'points_desain': d_d[i]['points']})
                else:
                    for item in d_o:
                        match = next((x for x in d_d if x['STA'] == item['STA']), None)
                        if match:
                            c, f = hitung_cut_fill(item['points'], match['points'])
                            res.append({'STA': item['STA'], 'cut': c, 'fill': f, 'points_tanah': item['points'], 'points_desain': match['points']})
                
                if res:
                    st.session_state['res_cross'] = res
                    st.success(f"Sukses: {len(res)} Data.")

    with col2:
        if 'res_cross' in st.session_state:
            res = st.session_state['res_cross']
            df_show = pd.DataFrame(res)[['STA', 'cut', 'fill']]
            st.dataframe(df_show, use_container_width=True, height=200)
            
            # Preview
            if res:
                item = res[0]
                fig, ax = plt.subplots(figsize=(8, 2))
                tx, ty = zip(*item['points_tanah'])
                dx, dy = zip(*item['points_desain'])
                ax.plot(tx, ty, 'k--', label='Tanah')
                ax.plot(dx, dy, 'r-', label='Desain')
                ax.fill_between(tx, ty, min(ty)-2, color='gray', alpha=0.1)
                ax.set_title(f"Preview {item['STA']}")
                st.pyplot(fig)
                
                dxf_data = generate_dxf_cross(res)
                st.download_button("📥 Download DXF Cross", dxf_data, "Cross_Section.dxf", "application/dxf")

# ==========================================
# TAB 2: LONG SECTION (New)
# ==========================================
with tab_long:
    st.header("Analisis Potongan Memanjang")
    st.info("Input file Excel PCLP Long Section (Biasanya format X=Jarak Kumulatif, Y=Elevasi)")
    
    col_l1, col_l2 = st.columns([1, 2])
    
    with col_l1:
        f_long = st.file_uploader("Upload Excel Long (.xls/.xlsx)", key="u_long")
        if f_long:
            xls_l = pd.ExcelFile(f_long)
            sl_ogl = st.selectbox("Sheet Tanah (Long):", xls_l.sheet_names, key="sl_ogl")
            sl_des = st.selectbox("Sheet Desain (Long):", xls_l.sheet_names, index=min(1,len(xls_l.sheet_names)-1), key="sl_des")
            
            if st.button("Proses Long Section", key="b_long"):
                # Parsing
                dfl_o = pd.read_excel(f_long, sheet_name=sl_ogl, header=None)
                dfl_d = pd.read_excel(f_long, sheet_name=sl_des, header=None)
                
                parsed_lo = parse_pclp_smart(dfl_o)
                parsed_ld = parse_pclp_smart(dfl_d)
                
                # MERGE LOGIC: Data Long biasanya terpecah per lembar di Excel
                # Kita gabungkan semua titik menjadi satu garis panjang
                merged_o = []
                for p in parsed_lo: merged_o.extend(p['points'])
                # Sort by X (Distance) agar garis nyambung rapi
                merged_o.sort(key=lambda x: x[0])
                
                merged_d = []
                for p in parsed_ld: merged_d.extend(p['points'])
                merged_d.sort(key=lambda x: x[0])
                
                st.session_state['long_res'] = (merged_o, merged_d)
                st.success(f"Terbaca: Tanah {len(merged_o)} titik, Desain {len(merged_d)} titik.")

    with col_l2:
        if 'long_res' in st.session_state:
            mo, md = st.session_state['long_res']
            
            # Plot Matplotlib
            fig, ax = plt.subplots(figsize=(10, 3))
            if mo:
                x, y = zip(*mo)
                ax.plot(x, y, 'k-', linewidth=1, label='Tanah Asli')
                ax.fill_between(x, y, min(y)-5, color='gray', alpha=0.1)
            if md:
                xd, yd = zip(*md)
                ax.plot(xd, yd, 'r-', linewidth=2, label='Desain')
            
            ax.set_title("Profil Memanjang (Long Section)")
            ax.set_xlabel("Jarak (m)")
            ax.set_ylabel("Elevasi (m)")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
            # Download DXF
            dxf_long = generate_dxf_long(mo, md)
            st.download_button("📥 Download DXF Long Section", dxf_long, "Long_Section.dxf", "application/dxf")

# ==========================================
# TAB 3: GLOBAL MAPPER TOOLS (New)
# ==========================================
with tab_gis:
    st.header("Import Data Geospasial (Global Mapper)")
    
    if not HAS_GEO_LIBS:
        st.error("⚠️ Library Geospasial (geopandas, rasterio) belum terinstall di server ini.")
        st.code("pip install geopandas rasterio", language="bash")
    else:
        st.markdown("""
        **Fungsi:** Mengambil data elevasi (sampling) dari file DEM/Raster berdasarkan garis Alinyemen (Trase).
        Hasilnya dapat didownload sebagai CSV untuk diolah menjadi Cross/Long Section.
        """)
        
        c_g1, c_g2 = st.columns(2)
        
        with c_g1:
            st.subheader("1. Input Data")
            f_dem = st.file_uploader("Upload DEM/DTM (.tif)", type=["tif", "tiff"])
            f_align = st.file_uploader("Upload Alinyemen (.geojson)", type=["json", "geojson"])
            st.caption("*Saran: Export As/Trase dari Global Mapper ke format GeoJSON agar mudah dibaca.*")
            
            step = st.number_input("Interval Sampling (meter):", value=50, min_value=1)
            
        with c_g2:
            st.subheader("2. Hasil Sampling")
            if f_dem and f_align:
                if st.button("Generate Long Section Data"):
                    with st.spinner("Sampling Raster..."):
                        # Simpan file sementara (Rasterio butuh path fisik biasanya, atau MemoryFile)
                        # Untuk Streamlit Cloud, kita pakai MemoryFile logic di process function atau tempfile
                        # Sederhananya kita passing object file-like buffer
                        pts, err = process_spatial_data(f_dem, f_align, step)
                        
                        if err:
                            st.error(f"Error: {err}")
                        else:
                            st.success(f"Berhasil mengambil {len(pts)} titik elevasi!")
                            
                            # Convert to DataFrame
                            df_gis = pd.DataFrame(pts, columns=["Jarak (X)", "Elevasi (Y)"])
                            st.dataframe(df_gis, height=200)
                            
                            # Plot Preview
                            st.line_chart(df_gis.set_index("Jarak (X)"))
                            
                            # Download CSV
                            csv = df_gis.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download CSV Hasil Sampling", csv, "Sampling_GlobalMapper.csv", "text/csv")
            else:
                st.info("Silakan upload file DEM (.tif) dan Garis Trase (.geojson) terlebih dahulu.")
