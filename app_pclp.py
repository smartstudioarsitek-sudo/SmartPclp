import streamlit as st
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, LineString
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
    # Loop baris demi baris
    while i < len(df):
        row = df.iloc[i]
        x_col_idx = -1
        max_col_check = min(15, len(row))
        
        # Scan posisi 'X' di baris ini
        for c in range(max_col_check):
            try:
                if str(row[c]).strip().upper() == 'X':
                    x_col_idx = c
                    break
            except: continue
        
        # Jika ketemu X, cek baris bawahnya apakah 'Y'
        if x_col_idx != -1 and (i + 1 < len(df)):
            try:
                if str(df.iloc[i+1][x_col_idx]).strip().upper() == 'Y':
                    # --- KETEMU BLOCK DATA ---
                    
                    # Ambil Nama (STA) - biasanya 2 kolom sebelum X
                    sta_name = f"STA_{i}"
                    if x_col_idx >= 2:
                        val_sta = str(df.iloc[i+1][x_col_idx - 2]).strip()
                        if val_sta and val_sta.lower() != 'nan': sta_name = val_sta
                    if sta_name.endswith('.0'): sta_name = sta_name[:-2]

                    # Ambil Data Angka (Mulai kolom setelah X)
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
                    
                    i += 1 # Lompat baris Y
            except: pass
        i += 1
    return parsed_data

def hitung_cut_fill(tanah_pts, desain_pts):
    """Menghitung luas area Cut & Fill per cross section."""
    if not tanah_pts or not desain_pts: return 0, 0
    
    # Datum bantu agar polygon tertutup
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
        
        # Text Alignment Fix
        msp.add_mtext(info, dxfattribs={'char_height':0.5, 'layer':'TEKS_DATA'}).set_location(
            insert=(cx, my+3), attachment_point=ezdxf.const.MTEXT_TOP_CENTER)
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
    
    # Label
    if merged_tanah:
        min_x = min(p[0] for p in merged_tanah)
        min_y = min(p[1] for p in merged_tanah)
        msp.add_text("LONG SECTION PROFILE", dxfattribs={'height': 2.0}).set_placement((min_x, min_y-10))

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
        if gdf.crs and gdf.crs.is_geographic:
            # Estimasi ke Pseudo-Mercator untuk hitungan meter
            gdf = gdf.to_crs(epsg=3857) 
        
        line_geom = gdf.geometry.iloc[0] 
        if not isinstance(line_geom, LineString):
            return None, "File GeoJSON harus berisi tipe LineString (Garis)!"

        # 2. Baca DEM
        # Gunakan rasterio MemoryFile jika input berupa bytes dari Streamlit
        # (Streamlit Cloud butuh penanganan khusus, ini versi simplenya)
        with rasterio.open(dem_file) as src:
            # Sampling logic
            distances = np.arange(0, line_geom.length, step_m)
            long_points = []
            
            for dist in distances:
                pt = line_geom.interpolate(dist)
                # Note: Coordinate transformation might be needed here depending on CRS match
                # Assuming CRS matches for this demo
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
st.set_page_config(page_title="PCLP Studio v2", layout="wide")
st.title("🚜 PCLP Studio: Engineering Suite")

# Cek Library
if not HAS_GEO_LIBS:
    st.warning("⚠️ Mode Terbatas: Fitur Global Mapper non-aktif. Install `geopandas rasterio` untuk mengaktifkan.")

# --- TABS MENU ---
tab_cross, tab_long, tab_gis = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🌍 GLOBAL MAPPER"])

# ==========================================
# TAB 1: CROSS SECTION
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
# TAB 2: LONG SECTION (New Feature)
# ==========================================
with tab_long:
    st.header("Analisis Potongan Memanjang")
    st.info("Fitur ini menggabungkan data per halaman menjadi satu grafik panjang.")
    
    
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
                
                # MERGE LOGIC: Menggabungkan titik-titik menjadi satu array panjang
                merged_o = []
                for p in parsed_lo: merged_o.extend(p['points'])
                merged_o.sort(key=lambda x: x[0]) # Urutkan berdasarkan Jarak (X)
                
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
            ax.set_xlabel("Jarak Kumulatif (m)")
            ax.set_ylabel("Elevasi (m)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
            # Download DXF
            dxf_long = generate_dxf_long(mo, md)
            st.download_button("📥 Download DXF Long Section", dxf_long, "Long_Section.dxf", "application/dxf")

# ==========================================
# TAB 3: GLOBAL MAPPER TOOLS (New Feature)
# ==========================================
with tab_gis:
    st.header("Import Data Geospasial")
    
    
    if not HAS_GEO_LIBS:
        st.error("⚠️ Library `geopandas` & `rasterio` belum terinstall.")
    else:
        st.markdown("**Sampling DEM otomatis:** Upload DEM (.tif) dan Garis Trase (.geojson) untuk mendapatkan data Long Section.")
        
        c_g1, c_g2 = st.columns(2)
        
        with c_g1:
            st.subheader("1. Input Data")
            f_dem = st.file_uploader("Upload DEM/DTM (.tif)", type=["tif", "tiff"])
            f_align = st.file_uploader("Upload Alinyemen (.geojson)", type=["json", "geojson"])
            
            step = st.number_input("Interval Sampling (meter):", value=50, min_value=1)
            
        with c_g2:
            st.subheader("2. Hasil")
            if f_dem and f_align:
                if st.button("Generate Sampling"):
                    with st.spinner("Sedang memproses..."):
                        pts, err = process_spatial_data(f_dem, f_align, step)
                        
                        if err:
                            st.error(f"Error: {err}")
                        else:
                            st.success(f"Berhasil: {len(pts)} titik elevasi.")
                            df_gis = pd.DataFrame(pts, columns=["Jarak (X)", "Elevasi (Y)"])
                            st.line_chart(df_gis.set_index("Jarak (X)"))
                            
                            csv = df_gis.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download CSV", csv, "Sampling_GM.csv", "text/csv")
