import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from shapely.geometry import Polygon, LineString

# --- HANDLING IMPORT LIBRARY ---
# Menggunakan try-except agar aplikasi tidak mati total jika library kurang
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
except ImportError:
    st.warning("⚠️ Library 'ezdxf' belum terinstall. Fitur DXF tidak akan jalan.")

HAS_GEO_LIBS = False
try:
    import geopandas as gpd
    import rasterio
    from rasterio.plot import show
    HAS_GEO_LIBS = True
except ImportError:
    pass

# CATATAN: st.set_page_config SUDAH DIHAPUS 
# Karena file ini dipanggil oleh main.py, jadi tidak boleh ada config ganda.

# ==========================================
# 1. PARSER ENGINE (ROBUST)
# ==========================================
def parse_pclp_block(df):
    """Parser untuk format Excel Blok PCLP (Cross Section)."""
    parsed_data = []
    i = 0
    # Konversi semua ke string agar aman
    df = df.astype(str)
    
    while i < len(df):
        row = df.iloc[i].values
        # Cari huruf 'X' di baris ini sebagai penanda header
        x_indices = [idx for idx, val in enumerate(row) if val.strip().upper() == 'X']
        
        if x_indices and (i + 1 < len(df)):
            x_idx = x_indices[0] # Ambil 'X' pertama yang ketemu
            
            # Cek apakah baris bawahnya ada 'Y'
            val_y = str(df.iloc[i+1, x_idx]).strip().upper()
            
            if val_y == 'Y':
                # Ambil Nama STA
                sta_name = f"STA_{len(parsed_data)}"
                
                # Coba cari nama STA di kolom ke-1 atau ke-2 baris Y
                candidate_sta = str(df.iloc[i+1, 1]).strip() 
                if candidate_sta.lower() not in ['nan', 'none', '']:
                    sta_name = candidate_sta
                
                # Bersihkan nama STA
                if sta_name.endswith('.0'): sta_name = sta_name[:-2]

                # Ekstraksi Data
                start_col = x_idx + 1
                row_x = df.iloc[i].values
                row_y = df.iloc[i+1].values
                
                points = []
                for c in range(start_col, len(row_x)):
                    try:
                        vx = float(str(row_x[c]).replace(',', '.'))
                        vy = float(str(row_y[c]).replace(',', '.'))
                        if not (math.isnan(vx) or math.isnan(vy)):
                            points.append((vx, vy))
                    except:
                        break # Berhenti jika ketemu non-angka
                
                if points:
                    points.sort(key=lambda p: p[0])
                    parsed_data.append({'STA': sta_name, 'points': points})
                
                i += 1 # Skip baris Y
        i += 1
    return parsed_data

def hitung_cut_fill(tanah_pts, desain_pts):
    """Menghitung luas Cut & Fill menggunakan Shapely."""
    if not tanah_pts or not desain_pts: 
        return 0.0, 0.0
    
    # Cari elevasi terendah untuk datum buatan
    min_y = min([p[1] for p in tanah_pts] + [p[1] for p in desain_pts])
    datum = min_y - 5.0
    
    # Buat Poligon Tanah
    p_tanah = tanah_pts + [(tanah_pts[-1][0], datum), (tanah_pts[0][0], datum)]
    poly_tanah = Polygon(p_tanah).buffer(0) # Buffer 0 fix self-intersection
    
    # Buat Poligon Desain
    p_desain = desain_pts + [(desain_pts[-1][0], datum), (desain_pts[0][0], datum)]
    poly_desain = Polygon(p_desain).buffer(0)
    
    try:
        area_cut = poly_desain.intersection(poly_tanah).area
        area_fill = poly_desain.difference(poly_tanah).area
    except:
        area_cut, area_fill = 0.0, 0.0
        
    return area_cut, area_fill

# ==========================================
# 2. GENERATOR OUTPUT (DXF & EXCEL)
# ==========================================
def generate_dxf(results, mode="cross"):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Setup Layers
    doc.layers.add(name='TANAH', color=8)   # Abu-abu
    doc.layers.add(name='DESAIN', color=1)  # Merah
    doc.layers.add(name='TEXT', color=7)    # Putih
    doc.layers.add(name='GRID', color=9)    # Abu Terang

    if mode == "long":
        tanah, desain = results
        # Gambar Long Section
        if tanah: msp.add_lwpolyline(tanah, dxfattribs={'layer': 'TANAH'})
        if desain: msp.add_lwpolyline(desain, dxfattribs={'layer': 'DESAIN'})
        
        # Tambah Grid Sederhana
        if tanah:
            min_x, max_x = min(p[0] for p in tanah), max(p[0] for p in tanah)
            min_y, max_y = min(p[1] for p in tanah), max(p[1] for p in tanah)
            msp.add_line((min_x, min_y), (max_x, min_y), dxfattribs={'layer': 'GRID'})
            msp.add_text("LONG SECTION PROFILE", dxfattribs={'height': 2.0, 'layer': 'TEXT'}).set_placement((min_x, max_y + 5))

    else: # Cross Section
        for i, item in enumerate(results):
            col = i % 2
            row = i // 2
            offset_x = col * 60  # Jarak antar kolom
            offset_y = row * -40 # Jarak antar baris
            
            t_pts = item.get('points_tanah', [])
            d_pts = item.get('points_desain', [])
            
            if t_pts:
                shifted_t = [(p[0] + offset_x, p[1] + offset_y) for p in t_pts]
                msp.add_lwpolyline(shifted_t, dxfattribs={'layer': 'TANAH'})
            
            if d_pts:
                shifted_d = [(p[0] + offset_x, p[1] + offset_y) for p in d_pts]
                msp.add_lwpolyline(shifted_d, dxfattribs={'layer': 'DESAIN'})
            
            center_x = offset_x
            base_y = offset_y + (min([p[1] for p in t_pts]) if t_pts else 0) - 2
            
            info_txt = f"{item['STA']}"
            if t_pts and d_pts:
                info_txt += f" | C:{item['cut']:.2f} | F:{item['fill']:.2f}"
            
            msp.add_text(info_txt, dxfattribs={'height': 0.5, 'layer': 'TEXT'}).set_placement((center_x, base_y))

    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

def generate_excel_report(data):
    """Membuat laporan Excel rekap volume."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        rekap = []
        for item in data:
            rekap.append({
                'STA': item['STA'],
                'Cut Area (m2)': item['cut'],
                'Fill Area (m2)': item['fill']
            })
        df = pd.DataFrame(rekap)
        df.to_excel(writer, sheet_name='Volume Report', index=False)
    return output.getvalue()

# ==========================================
# 3. GEOSPATIAL ENGINE (SITUASI)
# ==========================================
def render_peta_situasi(dem_file, shp_file):
    if not HAS_GEO_LIBS:
        return None, "Library GIS tidak terinstall."
    
    try:
        with rasterio.open(dem_file) as src:
            gdf = gpd.read_file(shp_file)
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Downsample agar cepat
            data = src.read(1, out_shape=(src.height // 5, src.width // 5))
            
            # Masking NoData
            data_masked = np.ma.masked_where(data == src.nodata, data)
            
            # Grid Koordinat
            x = np.linspace(src.bounds.left, src.bounds.right, data.shape[1])
            y = np.linspace(src.bounds.top, src.bounds.bottom, data.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Kontur
            contours = ax.contour(X, Y, data_masked, levels=20, cmap='terrain', linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=6, fmt='%1.0f')
            
            # Trase
            gdf.plot(ax=ax, color='red', linewidth=2, label='Trase Jalan', zorder=5)
            
            # Label STA
            line = gdf.geometry.iloc[0]
            length = line.length
            for dist in np.arange(0, length, 50):
                pt = line.interpolate(dist)
                ax.plot(pt.x, pt.y, 'ko', markersize=3)
                ax.annotate(f"STA {int(dist)}", (pt.x, pt.y), xytext=(5, 5), textcoords='offset points', fontsize=7, color='black')

            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title("Peta Situasi: Kontur & Trase")
            return fig, None

    except Exception as e:
        return None, str(e)

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🚜 PCLP Studio Pro v6.1 (Stable)")
st.caption("Aplikasi Desain Irigasi & Jalan: Cross Section, Long Section & GIS Situasi")

if not HAS_GEO_LIBS:
    st.warning("⚠️ Modul Geospasial tidak aktif. Tab GIS hanya simulasi.")

# Definisikan Tabs
tabs = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🗺️ PETA SITUASI (GIS)"])

# --- TAB 1: CROSS SECTION ---
with tabs[0]:
    col_in, col_view = st.columns([1, 2])
    
    with col_in:
        st.subheader("Input Data PCLP")
        f_upload = st.file_uploader("Upload Excel (.xls/.xlsx)", type=['xls', 'xlsx'], key='cross_up')
        
        if f_upload:
            try:
                xls = pd.ExcelFile(f_upload)
                sheet_ogl = st.selectbox("Sheet Tanah Asli", ["[Pilih Sheet]"] + xls.sheet_names)
                sheet_dsn = st.selectbox("Sheet Desain", ["[Pilih Sheet]"] + xls.sheet_names)
                
                if st.button("🚀 PROSES DATA CROSS"):
                    with st.spinner("Sedang memproses..."):
                        data_ogl = []
                        data_dsn = []
                        
                        if sheet_ogl != "[Pilih Sheet]":
                            df_ogl = pd.read_excel(f_upload, sheet_name=sheet_ogl, header=None)
                            data_ogl = parse_pclp_block(df_ogl)
                            
                        if sheet_dsn != "[Pilih Sheet]":
                            df_dsn = pd.read_excel(f_upload, sheet_name=sheet_dsn, header=None)
                            data_dsn = parse_pclp_block(df_dsn)
                        
                        final_data = []
                        max_len = max(len(data_ogl), len(data_dsn))
                        
                        for i in range(max_len):
                            t_item = data_ogl[i] if i < len(data_ogl) else None
                            d_item = data_dsn[i] if i < len(data_dsn) else None
                            
                            sta = t_item['STA'] if t_item else (d_item['STA'] if d_item else f"STA_{i}")
                            t_pts = t_item['points'] if t_item else []
                            d_pts = d_item['points'] if d_item else []
                            
                            cut, fill = hitung_cut_fill(t_pts, d_pts)
                            
                            final_data.append({
                                'STA': sta,
                                'points_tanah': t_pts,
                                'points_desain': d_pts,
                                'cut': cut,
                                'fill': fill
                            })
                        
                        st.session_state['data_cross'] = final_data
                        st.success(f"Berhasil memproses {len(final_data)} Cross Section!")
            except Exception as e:
                st.error(f"Error membaca file: {e}")

    with col_view:
        if 'data_cross' in st.session_state:
            data = st.session_state['data_cross']
            list_sta = [d['STA'] for d in data]
            selected_sta = st.select_slider("Pilih Station:", options=list_sta)
            idx = list_sta.index(selected_sta)
            item = data[idx]
            
            c1, c2 = st.columns(2)
            offset_x = c1.number_input("Geser X (m)", value=0.0, step=0.5, key='off_x')
            offset_y = c2.number_input("Geser Y (m)", value=0.0, step=0.5, key='off_y')
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            t_pts = [(p[0]+offset_x, p[1]+offset_y) for p in item['points_tanah']]
            if t_pts:
                ax.plot(*zip(*t_pts), 'k-o', label='Tanah', linewidth=1)
                ax.fill_between([p[0] for p in t_pts], [p[1] for p in t_pts], min([p[1] for p in t_pts])-2, color='gray', alpha=0.1)
                
            d_pts = item['points_desain']
            if d_pts:
                ax.plot(*zip(*d_pts), 'r-', label='Desain', linewidth=2)
            
            ax.set_title(f"{item['STA']} | C:{item['cut']:.2f} | F:{item['fill']:.2f}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            dxf_data = generate_dxf(data, mode="cross")
            st.download_button("📥 Download DXF", dxf_data, "Cross_Sections.dxf", "application/dxf")

# --- TAB 2: LONG SECTION ---
with tabs[1]:
    st.subheader("Long Section")
    f_long = st.file_uploader("Upload Long Section", type=['xls', 'xlsx', 'csv'], key='long_up')
    
    if f_long:
        try:
            if f_long.name.endswith('.csv'):
                df = pd.read_csv(f_long)
            else:
                df = pd.read_excel(f_long)
            
            # Ambil 2 kolom pertama angka
            df = df.select_dtypes(include=[np.number]).dropna()
            if df.shape[1] >= 2:
                pts = df.iloc[:, :2].values.tolist()
                pts.sort(key=lambda x: x[0])
                st.session_state['long_res'] = (pts, []) # Simpan sebagai Tuple (Tanah, Desain Kosong)
                st.success("Data Long Section terbaca!")
            else:
                st.error("Tidak ditemukan kolom angka yang valid.")
        except Exception as e:
            st.error(f"Gagal baca file: {e}")

    if 'long_res' in st.session_state:
        ogl, _ = st.session_state['long_res']
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(*zip(*ogl), 'k--', label='Tanah Asli')
        ax.grid(True)
        st.pyplot(fig)
        
        dxf_long = generate_dxf((ogl, []), mode="long")
        st.download_button("📥 Download DXF Long", dxf_long, "Long_Section.dxf", "application/dxf")

# --- TAB 3: PETA SITUASI ---
with tabs[2]:
    st.header("🗺️ Peta Situasi")
    c1, c2 = st.columns([1, 3])
    with c1:
        up_dem = st.file_uploader("Upload DEM (.tif)", type=['tif', 'tiff'])
        up_shp = st.file_uploader("Upload Trase (.geojson/.shp)", type=['geojson', 'shp'], accept_multiple_files=True)
        
        shp_file = None
        if up_shp:
            for f in up_shp:
                if f.name.endswith('.geojson') or f.name.endswith('.shp'):
                    shp_file = f
                    break
        
        if up_dem and shp_file and st.button("RENDER PETA"):
            st.session_state['gis_files'] = (up_dem, shp_file)

    with c2:
        if 'gis_files' in st.session_state:
            dem, shp = st.session_state['gis_files']
            dem.seek(0); shp.seek(0)
            
            with st.spinner("Merender..."):
                fig, err = render_peta_situasi(dem, shp)
                if fig: st.pyplot(fig)
                else: st.error(err)
