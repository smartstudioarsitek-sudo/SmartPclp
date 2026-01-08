import streamlit as st
import pandas as pd
import ezdxf
from shapely.geometry import Polygon, LineString, Point, shape
import matplotlib.pyplot as plt
import io
import numpy as np
import math
import zipfile
import tempfile
import os

# Konfigurasi Halaman
st.set_page_config(page_title="PCLP Studio Pro v7.0 GIS", layout="wide", page_icon="🛰️")

# --- LIBRARY LOADING ---
HAS_GEO_LIBS = False
try:
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from rasterio.transform import xy
    from skimage import measure # Untuk algoritma kontur
    import folium
    from streamlit_folium import st_folium, folium_static
    from folium.plugins import Draw
    HAS_GEO_LIBS = True
except ImportError as e:
    st.error(f"Error Library: {e}")

# ==========================================
# 1. CORE ENGINE: PARSER & MATH
# ==========================================
def parse_pclp_block(df):
    """Parser PCLP Legacy (tetap dipertahankan untuk kompatibilitas)."""
    parsed_data = []
    i = 0
    df = df.astype(str)
    while i < len(df):
        row = df.iloc[i].values
        x_indices = [idx for idx, val in enumerate(row) if val.strip().upper() == 'X']
        if x_indices and (i + 1 < len(df)):
            x_idx = x_indices[0]
            val_y = str(df.iloc[i+1, x_idx]).strip().upper()
            if val_y == 'Y':
                sta_name = f"STA_{len(parsed_data)}"
                candidate_sta = str(df.iloc[i+1, 1]).strip() 
                if candidate_sta.lower() not in ['nan', 'none', '']:
                    sta_name = candidate_sta
                if sta_name.endswith('.0'): sta_name = sta_name[:-2]

                start_col = x_idx + 1
                row_x = df.iloc[i].values; row_y = df.iloc[i+1].values
                points = []
                for c in range(start_col, len(row_x)):
                    try:
                        vx = float(str(row_x[c]).replace(',', '.'))
                        vy = float(str(row_y[c]).replace(',', '.'))
                        if not (math.isnan(vx) or math.isnan(vy)):
                            points.append((vx, vy))
                    except: break
                if points:
                    points.sort(key=lambda p: p[0])
                    parsed_data.append({'STA': sta_name, 'points': points})
                i += 1
        i += 1
    return parsed_data

def hitung_cut_fill(tanah_pts, desain_pts):
    if not tanah_pts or not desain_pts: return 0.0, 0.0
    min_y = min([p[1] for p in tanah_pts] + [p[1] for p in desain_pts])
    datum = min_y - 5.0
    p_tanah = tanah_pts + [(tanah_pts[-1][0], datum), (tanah_pts[0][0], datum)]
    p_desain = desain_pts + [(desain_pts[-1][0], datum), (desain_pts[0][0], datum)]
    try:
        poly_tanah = Polygon(p_tanah).buffer(0)
        poly_desain = Polygon(p_desain).buffer(0)
        return poly_desain.intersection(poly_tanah).area, poly_desain.difference(poly_tanah).area
    except: return 0.0, 0.0

# ==========================================
# 2. GIS ENGINE: KMZ & CONTOURS
# ==========================================
def read_kmz_to_gdf(uploaded_file):
    """Mengekstrak KML dari KMZ dan membaca sebagai GeoDataFrame."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simpan KMZ sementara
            kmz_path = os.path.join(temp_dir, "temp.kmz")
            with open(kmz_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Unzip
            with zipfile.ZipFile(kmz_path, 'r') as z:
                kml_file = [x for x in z.namelist() if x.endswith(".kml")][0]
                z.extract(kml_file, temp_dir)
                kml_path = os.path.join(temp_dir, kml_file)
                
                # Baca dengan GeoPandas (fiona support KML)
                gdf = gpd.read_file(kml_path)
                return gdf
    except Exception as e:
        st.error(f"Gagal baca KMZ: {e}")
        return None

def generate_contours_vector(dem_src, interval=1.0):
    """
    Algoritma Marching Squares untuk menghasilkan garis kontur vektor dari Raster.
    Output: List of Dictionary {'level': float, 'geometry': LineString}
    """
    # Baca data raster
    arr = dem_src.read(1)
    transform = dem_src.transform
    
    # Handle NoData
    if dem_src.nodata is not None:
        arr = np.where(arr == dem_src.nodata, np.nan, arr)
    
    # Tentukan level kontur
    min_z = np.nanmin(arr)
    max_z = np.nanmax(arr)
    levels = np.arange(np.floor(min_z), np.ceil(max_z), interval)
    
    contours_vectors = []
    
    for level in levels:
        # skimage returns list of (row, col) coordinates
        contours = measure.find_contours(arr, level)
        
        for contour in contours:
            # Transform pixel (row, col) ke world (x, y)
            # Perhatikan: contour[:, 1] adalah col (x), contour[:, 0] adalah row (y)
            cols = contour[:, 1]
            rows = contour[:, 0]
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
            if len(xs) > 1:
                pts = list(zip(xs, ys))
                line = LineString(pts)
                contours_vectors.append({'level': level, 'geometry': line})
                
    return contours_vectors

def create_dxf_from_gis(contours_data, trase_gdf=None):
    """Membuat DXF 3D dari data kontur dan trase."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Layers
    doc.layers.add(name='KONTUR_MAYOR', color=1) # Merah
    doc.layers.add(name='KONTUR_MINOR', color=7) # Putih
    doc.layers.add(name='TRASE_JALAN', color=3)  # Hijau
    doc.layers.add(name='TEXT_ELEVASI', color=2) # Kuning

    # Gambar Kontur
    for c in contours_data:
        lvl = c['level']
        is_major = (lvl % 5 == 0) # Interval mayor tiap 5m
        layer = 'KONTUR_MAYOR' if is_major else 'KONTUR_MINOR'
        
        # Konversi LineString ke List points
        pts = list(c['geometry'].coords)
        
        # Tambahkan LWPOLYLINE dengan Elevasi (Fitur 3D AutoCAD)
        msp.add_lwpolyline(pts, dxfattribs={
            'layer': layer,
            'elevation': lvl # Ini kuncinya agar terbaca 3D
        })
    
    # Gambar Trase (Jika ada)
    if trase_gdf is not None:
        for idx, row in trase_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'LineString':
                pts = list(geom.coords)
                msp.add_lwpolyline(pts, dxfattribs={'layer': 'TRASE_JALAN', 'lineweight': 30})
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    pts = list(line.coords)
                    msp.add_lwpolyline(pts, dxfattribs={'layer': 'TRASE_JALAN', 'lineweight': 30})

    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

# ==========================================
# 3. UI GENERATORS
# ==========================================
def generate_dxf_cross(results):
    """DXF Generator untuk Cross Section (Legacy Code)."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add(name='TANAH', color=8); doc.layers.add(name='DESAIN', color=1); doc.layers.add(name='TEXT', color=7)
    
    for i, item in enumerate(results):
        col, row = i % 2, i // 2
        off_x, off_y = col * 60, row * -40
        
        t_pts = [(p[0]+off_x, p[1]+off_y) for p in item.get('points_tanah', [])]
        d_pts = [(p[0]+off_x, p[1]+off_y) for p in item.get('points_desain', [])]
        
        if t_pts: msp.add_lwpolyline(t_pts, dxfattribs={'layer': 'TANAH'})
        if d_pts: msp.add_lwpolyline(d_pts, dxfattribs={'layer': 'DESAIN'})
        
        info = f"{item['STA']} | C:{item['cut']:.2f} | F:{item['fill']:.2f}"
        msp.add_text(info, dxfattribs={'height': 0.5, 'layer': 'TEXT'}).set_placement((off_x, off_y-2))
        
    out = io.StringIO(); doc.write(out)
    return out.getvalue().encode('utf-8')

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================
st.title("🛰️ PCLP Studio Pro v7.0 (GIS Revolution)")
st.markdown("""
**Fitur Baru:** 1. Import **KMZ Google Earth** & Interactive Drawing.
2. Auto-Generate **Kontur DXF** dari DEM.
3. Integrasi **Geospasial** ke Desain Teknik.
""")

tabs = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🌍 GIS & TOPOGRAFI (NEW)"])

# --- TAB 1: CROSS SECTION ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    with col1:
        f_cross = st.file_uploader("Upload Excel PCLP", type=['xls', 'xlsx'], key='up_cross')
        if f_cross:
            xls = pd.ExcelFile(f_cross)
            s_ogl = st.selectbox("Sheet Tanah", xls.sheet_names, key='s1')
            s_dsn = st.selectbox("Sheet Desain", xls.sheet_names, key='s2')
            if st.button("Proses Cross"):
                d_ogl = parse_pclp_block(pd.read_excel(f_cross, sheet_name=s_ogl, header=None))
                d_dsn = parse_pclp_block(pd.read_excel(f_cross, sheet_name=s_dsn, header=None))
                final = []
                for i in range(max(len(d_ogl), len(d_dsn))):
                    to = d_ogl[i] if i < len(d_ogl) else None
                    td = d_dsn[i] if i < len(d_dsn) else None
                    tp = to['points'] if to else []
                    dp = td['points'] if td else []
                    c, f = hitung_cut_fill(tp, dp)
                    final.append({'STA': to['STA'] if to else f"STA_{i}", 'points_tanah': tp, 'points_desain': dp, 'cut': c, 'fill': f})
                st.session_state['res_cross'] = final
                st.success("Selesai!")

    with col2:
        if 'res_cross' in st.session_state:
            res = st.session_state['res_cross']
            sel = st.select_slider("Pilih STA", options=[r['STA'] for r in res])
            dat = next(item for item in res if item['STA'] == sel)
            
            fig, ax = plt.subplots(figsize=(8,3))
            if dat['points_tanah']: ax.plot(*zip(*dat['points_tanah']), 'k-o', label='Tanah')
            if dat['points_desain']: ax.plot(*zip(*dat['points_desain']), 'r-', label='Desain')
            ax.set_title(f"Cut: {dat['cut']:.2f} m2 | Fill: {dat['fill']:.2f} m2")
            ax.legend(); ax.grid(True)
            st.pyplot(fig)
            
            st.download_button("📥 DXF Cross", generate_dxf_cross(res), "Cross.dxf")

# --- TAB 2: LONG SECTION ---
with tabs[1]:
    st.info("Fitur Long Section standard (Upload Excel/CSV Jarak & Elevasi)")
    # (Kode Long Section disederhanakan untuk fokus ke GIS, gunakan kode lama jika perlu)
    f_long = st.file_uploader("Upload CSV Long", type=['csv'])
    if f_long:
        df = pd.read_csv(f_long).dropna()
        st.line_chart(df.iloc[:, 1])

# --- TAB 3: GIS & TOPOGRAFI (CORE UPDATE) ---
with tabs[2]:
    st.header("🌍 Analisis Topografi & Konversi CAD")
    
    col_map, col_proc = st.columns([1.5, 1])
    
    # 1. INPUT INTERFACE
    with col_map:
        st.subheader("1. Area of Interest (AOI)")
        
        # Pilihan Input
        input_mode = st.radio("Metode Input:", ["Upload KMZ/KML", "Gambar Manual di Peta"])
        
        aoi_gdf = None
        
        if input_mode == "Upload KMZ/KML":
            f_kmz = st.file_uploader("Upload File Google Earth (.kmz/.kml)", type=['kmz', 'kml'])
            if f_kmz:
                aoi_gdf = read_kmz_to_gdf(f_kmz)
                if aoi_gdf is not None:
                    st.success(f"Berhasil memuat {len(aoi_gdf)} fitur geometri!")
        
        # Inisialisasi Peta
        m = folium.Map(location=[-5.3971, 105.2668], zoom_start=10) # Default Lampung
        
        # Jika ada AOI dari KMZ, tambahkan ke peta
        if aoi_gdf is not None:
            # Reproject ke WGS84 untuk Folium
            aoi_web = aoi_gdf.to_crs("EPSG:4326")
            folium.GeoJson(aoi_web).add_to(m)
            # Zoom ke area
            bounds = aoi_web.total_bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Draw Control
        draw = Draw(
            export=True,
            filename='my_data.geojson',
            draw_options={'polyline': True, 'polygon': True, 'rectangle': True, 'circle': False, 'marker': False}
        )
        draw.add_to(m)
        
        # Render Peta
        output = st_folium(m, width=700, height=500)
        
        # Tangkap Hasil Gambar Manual
        if input_mode == "Gambar Manual di Peta" and output['all_drawings']:
            # Konversi JSON gambar ke GeoDataFrame
            drawings = output['all_drawings']
            if drawings:
                features = [d['geometry'] for d in drawings]
                # Buat GDF dummy
                # Ini logic simple, idealnya parse GeoJSON full
                from shapely.geometry import shape
                geoms = [shape(f) for f in features]
                aoi_gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
                st.info("Sketsa manual terdeteksi.")

    # 2. PROCESSING INTERFACE
    with col_proc:
        st.subheader("2. Pemrosesan Data DEM")
        
        st.warning("⚠️ Upload DEM (GeoTIFF) dari SRTM/Demnas/Copernicus.")
        f_dem = st.file_uploader("Upload File DEM (.tif)", type=['tif', 'tiff'])
        
        if f_dem and aoi_gdf is not None:
            # Tampilkan tombol proses
            interval = st.number_input("Interval Kontur (m)", min_value=1.0, value=5.0, step=1.0)
            
            if st.button("⚙️ GENERATE KONTUR & DXF"):
                with st.spinner("Sedang memproses topografi..."):
                    try:
                        # 1. Buka Raster
                        with rasterio.open(f_dem) as src:
                            # 2. Clip Raster sesuai AOI (Opsional, di sini kita baca full dulu untuk demo)
                            # Idealnya menggunakan mask.mask
                            
                            # Cek CRS AOI, samakan dengan Raster
                            aoi_proj = aoi_gdf.to_crs(src.crs)
                            
                            # Masking/Clipping
                            out_image, out_transform = rasterio.mask.mask(src, aoi_proj.geometry, crop=True)
                            out_meta = src.meta.copy()
                            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
                            
                            # Simpan Memory File untuk Contour Engine
                            with rasterio.io.MemoryFile() as memfile:
                                with memfile.open(**out_meta) as dataset:
                                    dataset.write(out_image)
                                    
                                    # 3. Generate Vector Contours
                                    contours = generate_contours_vector(dataset, interval)
                            
                            st.success(f"Berhasil membuat {len(contours)} segmen kontur!")
                            
                            # 4. Preview Plot Simple
                            fig, ax = plt.subplots()
                            for c in contours:
                                x, y = c['geometry'].xy
                                ax.plot(x, y, linewidth=0.5, color='brown')
                            # Plot Trase/AOI
                            aoi_proj.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2)
                            st.pyplot(fig)
                            
                            # 5. Export DXF
                            dxf_bytes = create_dxf_from_gis(contours, aoi_proj)
                            st.download_button(
                                label="📥 DOWNLOAD CAD (DXF 3D)",
                                data=dxf_bytes,
                                file_name="Kontur_Situasi.dxf",
                                mime="application/dxf"
                            )
                            
                    except Exception as e:
                        st.error(f"Error Processing: {str(e)}")
                        st.write("Tips: Pastikan CRS DEM dan Lokasi AOI beririsan.")
        
        elif not f_dem:
            st.info("Menunggu upload DEM...")
        elif aoi_gdf is None:
            st.info("Silakan upload KMZ atau gambar area di peta dulu.")
