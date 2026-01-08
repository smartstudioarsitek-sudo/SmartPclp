# ==============================================================================
# SMART PCLP INTEGRATED SYSTEM (HYDRAULIC + CIVIL DESIGN)
# Gabungan Fitur: Analisis Hidrologi (Lama) + Cross Section/Long Section (Baru)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import math
import matplotlib.pyplot as plt

# --- LIBRARY HIDROLOGI & GIS ---
try:
    import geopandas as gpd
    import rasterio
    from rasterio import MemoryFile
    import rioxarray
    from rioxarray.merge import merge_arrays
    import pystac_client
    import planetary_computer
    from pysheds.grid import Grid
    import leafmap.foliumap as leafmap
    from streamlit_folium import st_folium
    from shapely.geometry import Polygon, LineString, Point
except ImportError as e:
    st.error(f"Library Geospasial Kurang: {e}. Pastikan requirements.txt lengkap.")

# --- LIBRARY TEKNIK SIPIL (DXF) ---
try:
    import ezdxf
except ImportError:
    st.warning("Library ezdxf belum terinstall. Fitur DXF mungkin tidak jalan.")

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Smart PCLP Studio: Integrated", layout="wide", page_icon="🌊")

# ==============================================================================
# BAGIAN 1: MESIN HIDROLOGI (FITUR LAMA)
# ==============================================================================

class HydroEngine:
    """Mesin untuk memproses DEM dan Delineasi DAS (Fitur Lama)"""
    def __init__(self, dem_path):
        # Membaca file DEM fisik (tempfile)
        self.grid = Grid.from_raster(dem_path)
        self.dem = self.grid.view('dem')

    def condition_dem(self):
        # Fill depressions / Pits
        self.grid.fill_depressions('dem', out_name='flooded_dem')
        self.grid.resolve_flats('flooded_dem', out_name='inflated_dem')
        
        # Flow Direction & Accumulation
        # N: North, NE: North-East, etc (D8 Routing)
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        self.grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
        self.grid.accumulation(data='dir', out_name='acc', dirmap=dirmap)
        return self.grid.view('acc')

    def delineate_catchment(self, x, y):
        # Snap Point ke sungai terdekat (High Accumulation)
        xy = (x, y)
        
        # Cari titik akumulasi tertinggi di sekitar klik (Radius kecil)
        # Note: Ini versi sederhana, idealnya snapping cari max acc
        snapped_xy = self.grid.snap_to_mask(self.grid.acc > 100, xy)
        
        # Delineasi
        self.grid.catchment(data='dir', x=snapped_xy[0], y=snapped_xy[1], 
                           dirmap=(64, 128, 1, 2, 4, 8, 16, 32), 
                           out_name='catch', recursionlimit=15000, xytype='coordinate')
        
        # Polygonize
        self.grid.clip_to('catch')
        shapes = self.grid.polygonize()
        
        # Ambil Polygon terbesar
        catchment_poly = None
        max_area = 0
        for shape, value in shapes:
            if value == 1: # Value mask catchment
                poly = Polygon(shape['coordinates'][0])
                if poly.area > max_area:
                    max_area = poly.area
                    catchment_poly = poly
                    
        return catchment_poly

# FUNGSI DOWNLOAD DEM (COPERNICUS)
@st.cache_data
def fetch_dem_copernicus(bbox):
    """Download DEM dari Microsoft Planetary Computer"""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox,
    )
    items = list(search.items())
    if not items: return None

    dem_arrays = []
    for item in items:
        signed_href = item.assets["data"].href
        da = rioxarray.open_rasterio(signed_href).rio.clip_box(*bbox)
        dem_arrays.append(da)
        
    if len(dem_arrays) > 1:
        dem_merged = merge_arrays(dem_arrays)
        return dem_merged
    return dem_arrays[0]

# ==============================================================================
# BAGIAN 2: MESIN SIPIL & PCLP (FITUR BARU)
# ==============================================================================

def parse_pclp_block(df):
    """Parser Excel PCLP Cross Section"""
    parsed_data = []
    i = 0
    df = df.astype(str)
    
    while i < len(df):
        row = df.iloc[i].values
        # Cari header X
        x_indices = [idx for idx, val in enumerate(row) if str(val).strip().upper() == 'X']
        
        if x_indices and (i + 1 < len(df)):
            x_idx = x_indices[0]
            val_y = str(df.iloc[i+1, x_idx]).strip().upper()
            
            if val_y == 'Y':
                # Ambil STA
                sta_name = f"STA_{len(parsed_data)}"
                candidate = str(df.iloc[i+1, 1]).strip()
                if candidate.lower() not in ['nan', 'none', '']:
                    sta_name = candidate.replace('.0', '')

                # Ambil Data X, Y
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
                    except: break
                
                if points:
                    points.sort(key=lambda p: p[0])
                    parsed_data.append({'STA': sta_name, 'points': points})
                i += 1
        i += 1
    return parsed_data

def hitung_cut_fill(tanah, desain):
    if not tanah or not desain: return 0.0, 0.0
    try:
        min_y = min([p[1] for p in tanah] + [p[1] for p in desain])
        datum = min_y - 5.0
        p_tanah = tanah + [(tanah[-1][0], datum), (tanah[0][0], datum)]
        p_desain = desain + [(desain[-1][0], datum), (desain[0][0], datum)]
        
        poly_t = Polygon(p_tanah).buffer(0)
        poly_d = Polygon(p_desain).buffer(0)
        
        return poly_d.intersection(poly_t).area, poly_d.difference(poly_t).area
    except: return 0.0, 0.0

def generate_dxf(results, mode="cross"):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add(name='TANAH', color=8)
    doc.layers.add(name='DESAIN', color=1)
    doc.layers.add(name='TEXT', color=7)

    if mode == "long":
        tanah, desain = results
        if tanah: msp.add_lwpolyline(tanah, dxfattribs={'layer': 'TANAH'})
        if desain: msp.add_lwpolyline(desain, dxfattribs={'layer': 'DESAIN'})
    else:
        for i, item in enumerate(results):
            col = i % 2
            row = i // 2
            ox, oy = col * 60, row * -40
            
            t_pts = [(p[0]+ox, p[1]+oy) for p in item.get('points_tanah', [])]
            d_pts = [(p[0]+ox, p[1]+oy) for p in item.get('points_desain', [])]
            
            if t_pts: msp.add_lwpolyline(t_pts, dxfattribs={'layer': 'TANAH'})
            if d_pts: msp.add_lwpolyline(d_pts, dxfattribs={'layer': 'DESAIN'})
            
            txt = f"{item['STA']} | C:{item['cut']:.2f} | F:{item['fill']:.2f}"
            msp.add_text(txt, dxfattribs={'height':0.5, 'layer':'TEXT'}).set_placement((ox, oy))
            
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

# ==============================================================================
# MAIN APPLICATION INTERFACE (SIDEBAR MENU)
# ==============================================================================

st.sidebar.title("🎛️ Menu Aplikasi")
menu_pilihan = st.sidebar.radio("Pilih Modul:", ["1. Analisis Hidrologi (DAS)", "2. Desain Sipil (Cross/Long)"])

# ------------------------------------------------------------------------------
# MODUL 1: ANALISIS HIDROLOGI (Code Lama yang dipulihkan)
# ------------------------------------------------------------------------------
if menu_pilihan == "1. Analisis Hidrologi (DAS)":
    st.title("💧 Analisis Hidrologi & Delineasi DAS")
    st.caption("Modul Analisis Topografi, Aliran Sungai, dan Catchment Area")

    # 1. INPUT WILAYAH (GEOJSON/KML)
    st.subheader("1. Tentukan Wilayah Studi")
    up_file = st.file_uploader("Upload Batas Wilayah (GeoJSON/KML/KMZ)", type=["geojson", "kml", "kmz", "json"])
    
    bbox = None
    if up_file:
        # Load File Logic (Simplified)
        import fiona
        try:
            # Fix Driver KML
            fiona.drvsupport.supported_drivers['KML'] = 'rw'
            fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
        except: pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_file.name.split('.')[-1]}") as tmp:
            tmp.write(up_file.getbuffer())
            tmp_path = tmp.name

        try:
            # Baca Data
            gdf = gpd.read_file(tmp_path)
            # Konversi ke WGS84 untuk Peta
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            bbox = gdf.total_bounds # [minx, miny, maxx, maxy]
            st.session_state['bbox'] = bbox
            st.success(f"Wilayah dimuat: {len(gdf)} fitur.")
        except Exception as e:
            st.error(f"Gagal baca file: {e}")

    # 2. DOWNLOAD & PROSES DEM
    if 'bbox' in st.session_state:
        st.subheader("2. Akuisisi Data DEM (Copernicus)")
        if st.button("⬇️ Download & Proses DEM"):
            with st.spinner("Mengunduh DEM dari Satelit..."):
                dem_data = fetch_dem_copernicus(st.session_state['bbox'])
                
                if dem_data is not None:
                    # Simpan DEM ke Tempfile untuk Pysheds (Wajib File Fisik)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f_dem:
                        dem_data.rio.to_raster(f_dem.name)
                        st.session_state['dem_path'] = f_dem.name
                    
                    st.success("DEM berhasil diunduh!")
                    
                    # Proses Hidrologi Awal
                    eng = HydroEngine(st.session_state['dem_path'])
                    acc = eng.condition_dem()
                    st.session_state['hydro_engine'] = eng # Simpan Objek Engine
                    st.success("Analisis Arah Aliran Selesai!")

    # 3. INTERAKTIF PETA & DELINEASI
    st.subheader("3. Peta Interaktif & Delineasi")
    
    if 'dem_path' in st.session_state:
        m = leafmap.Map()
        
        # Tambahkan DEM ke Peta
        m.add_raster(st.session_state['dem_path'], layer_name="DEM Topografi", colormap="terrain")
        
        # Tambahkan Sungai (Thresholding Accumulation)
        # Note: Ini visualisasi kasar sungai
        # (Idealnya konversi raster stream ke vector dulu, tapi untuk cepat kita pakai raster)
        
        st.write("klik pada peta di area sungai (lembah) untuk delineasi DAS.")
        
        # Render Peta dengan Folium
        map_out = st_folium(m, height=500, width=None)
        
        # Logika Klik untuk Delineasi
        if map_out['last_clicked']:
            lat = map_out['last_clicked']['lat']
            lng = map_out['last_clicked']['lng']
            st.info(f"Titik Outlet Dipilih: {lat}, {lng}")
            
            if 'hydro_engine' in st.session_state:
                with st.spinner("Menghitung Batas DAS (Delineasi)..."):
                    eng = st.session_state['hydro_engine']
                    poly_das = eng.delineate_catchment(lng, lat)
                    
                    if poly_das:
                        st.success(f"DAS Terhitung! Luas: {poly_das.area:.6f} deg²")
                        # Konversi ke GeoDataFrame untuk Download
                        gdf_das = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                        
                        # Download Button
                        json_das = gdf_das.to_json()
                        st.download_button("📥 Download GeoJSON DAS", json_das, "batas_das.geojson", "application/json")
                    else:
                        st.warning("Gagal mendelineasi. Coba klik lebih dekat ke alur sungai.")

# ------------------------------------------------------------------------------
# MODUL 2: DESAIN SIPIL (Fitur PCLP Baru)
# ------------------------------------------------------------------------------
elif menu_pilihan == "2. Desain Sipil (Cross/Long)":
    st.title("🚜 PCLP Desain Sipil")
    st.caption("Modul Cross Section, Long Section, dan Volume Cut/Fill")
    
    tab_cross, tab_long = st.tabs(["📐 Cross Section", "📈 Long Section"])
    
    # --- SUB-TAB CROSS SECTION ---
    with tab_cross:
        st.write("### Input Data Cross Section")
        f_up = st.file_uploader("Upload Excel PCLP (.xlsx)", type=['xlsx', 'xls'])
        
        if f_up:
            xls = pd.ExcelFile(f_up)
            s_ogl = st.selectbox("Sheet Tanah", xls.sheet_names, index=0)
            s_dsn = st.selectbox("Sheet Desain", xls.sheet_names, index=1 if len(xls.sheet_names)>1 else 0)
            
            if st.button("Proses Cross Section"):
                df_ogl = pd.read_excel(f_up, sheet_name=s_ogl, header=None)
                df_dsn = pd.read_excel(f_up, sheet_name=s_dsn, header=None)
                
                d_ogl = parse_pclp_block(df_ogl)
                d_dsn = parse_pclp_block(df_dsn)
                
                # Gabung & Hitung
                final = []
                for i in range(max(len(d_ogl), len(d_dsn))):
                    t = d_ogl[i] if i < len(d_ogl) else None
                    d = d_dsn[i] if i < len(d_dsn) else None
                    sta = t['STA'] if t else (d['STA'] if d else f"STA_{i}")
                    
                    tp = t['points'] if t else []
                    dp = d['points'] if d else []
                    c, f = hitung_cut_fill(tp, dp)
                    
                    final.append({'STA': sta, 'points_tanah': tp, 'points_desain': dp, 'cut': c, 'fill': f})
                
                st.session_state['cross_data'] = final
                st.success("Selesai Hitung!")
        
        if 'cross_data' in st.session_state:
            data = st.session_state['cross_data']
            idx = st.slider("Pilih STA", 0, len(data)-1, 0)
            item = data[idx]
            
            fig, ax = plt.subplots(figsize=(10, 3))
            if item['points_tanah']: ax.plot(*zip(*item['points_tanah']), 'k-o', label='Tanah')
            if item['points_desain']: ax.plot(*zip(*item['points_desain']), 'r-', label='Desain')
            ax.set_title(f"{item['STA']} (Cut: {item['cut']:.2f}, Fill: {item['fill']:.2f})")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            dxf = generate_dxf(data, "cross")
            st.download_button("📥 Download DXF Cross", dxf, "cross_section.dxf")

    # --- SUB-TAB LONG SECTION ---
    with tab_long:
        st.write("### Input Long Section")
        f_long = st.file_uploader("Upload Data Long (.csv/.xlsx)", type=['csv','xlsx'])
        if f_long:
            # Simple Reader Logic
            try:
                if f_long.name.endswith('.csv'):
                    df = pd.read_csv(f_long)
                else:
                    df = pd.read_excel(f_long)
                
                # Asumsi 2 kolom pertama: Jarak, Elevasi
                pts = df.iloc[:, :2].dropna().values.tolist()
                pts.sort(key=lambda x: x[0])
                
                st.session_state['long_pts'] = pts
                st.success("Data Long Section Terbaca!")
            except:
                st.error("Format file tidak sesuai.")
                
        if 'long_pts' in st.session_state:
            pts = st.session_state['long_pts']
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(*zip(*pts), 'b-')
            ax.set_title("Long Section Profile")
            ax.grid(True)
            st.pyplot(fig)
            
            # DXF
            dxf_long = generate_dxf((pts, []), "long")
            st.download_button("📥 Download DXF Long", dxf_long, "long_section.dxf")

# ==============================================================================
# END OF CODE
# ==============================================================================
