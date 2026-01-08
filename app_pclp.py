# ==============================================================================
# SMART PCLP STUDIO PRO - INTEGRATED VERSION
# Gabungan Fitur: Analisis Hidrologi (Fix) + Desain Sipil (Cross/Long Section)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import math
import matplotlib.pyplot as plt

# --- 1. IMPORT LIBRARY GEOSPASIAL & SIPIL ---
# Menggunakan try-except agar aplikasi tetap jalan meski library kurang (sebagai handling)
try:
    import geopandas as gpd
    import rasterio
    import rioxarray
    from rioxarray.merge import merge_arrays
    import pystac_client
    import planetary_computer
    from pysheds.grid import Grid
    import leafmap.foliumap as leafmap
    from streamlit_folium import st_folium
    from shapely.geometry import Polygon, LineString
    import fiona
except ImportError as e:
    st.error(f"⚠️ Error Import Library Geospasial: {e}. Cek requirements.txt Anda.")

try:
    import ezdxf
except ImportError:
    st.warning("⚠️ Library 'ezdxf' belum terinstall. Fitur export CAD tidak akan jalan.")

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Smart PCLP Studio v7.0", layout="wide", page_icon="🚜")

# ==============================================================================
# BAGIAN A: MESIN HIDROLOGI (HYDRO ENGINE - FIXED)
# ==============================================================================

class HydroEngine:
    """
    Mesin analisis hidrologi menggunakan Pysheds.
    Diperbaiki agar robust terhadap pembacaan metadata raster.
    """
    def __init__(self, dem_path):
        # 1. Inisialisasi Grid dari file fisik
        # Grid.from_raster otomatis membaca koordinat & proyeksi
        self.grid = Grid.from_raster(dem_path)
        
        # 2. Baca data elevasi (isi pixel) secara eksplisit
        dem_data = self.grid.read_raster(dem_path)
        
        # 3. Masukkan data ke dalam Grid
        # FIX: Kita tidak memaksa parameter 'affine'/'crs' manual agar tidak error.
        # Biarkan grid menggunakan metadata yang sudah dimuat di langkah 1.
        self.grid.add_gridded_data(dem_data, data_name='dem')
        
        # 4. Set view aktif ke 'dem'
        self.dem = self.grid.view('dem')

    def condition_dem(self):
        """Memperbaiki DEM (mengisi cekungan/pit filling)"""
        # Fill depressions
        self.grid.fill_depressions('dem', out_name='flooded_dem')
        self.grid.resolve_flats('flooded_dem', out_name='inflated_dem')
        
        # Flow Direction (D8) & Accumulation
        # Mapping arah: N, NE, E, SE, S, SW, W, NW
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        self.grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
        self.grid.accumulation(data='dir', out_name='acc', dirmap=dirmap)
        return self.grid.view('acc')

    def delineate_catchment(self, x, y):
        """Mendelineasi DAS dari titik koordinat (x, y)"""
        xy = (x, y)
        try:
            # Snap titik klik ke aliran sungai terdekat (Accumulation > 100)
            # Ini penting agar tidak delineasi di titik sembarang
            snapped_xy = self.grid.snap_to_mask(self.grid.acc > 100, xy)
            
            # Hitung Catchment (Daerah Tangkapan)
            self.grid.catchment(data='dir', x=snapped_xy[0], y=snapped_xy[1], 
                               dirmap=(64, 128, 1, 2, 4, 8, 16, 32), 
                               out_name='catch', recursionlimit=15000, xytype='coordinate')
            
            # Konversi Raster Catchment ke Polygon Vektor
            self.grid.clip_to('catch')
            shapes = self.grid.polygonize()
            
            # Cari polygon terbesar (DAS utama)
            catchment_poly = None
            max_area = 0
            for shape, value in shapes:
                if value == 1: # Nilai 1 adalah area catchment
                    poly = Polygon(shape['coordinates'][0])
                    if poly.area > max_area:
                        max_area = poly.area
                        catchment_poly = poly
                        
            return catchment_poly
        except Exception as e:
            st.error(f"Gagal Delineasi: {e}")
            return None

@st.cache_data
def fetch_dem_copernicus(bbox):
    """Fungsi Download DEM dari Microsoft Planetary Computer"""
    try:
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

        # Download & Merge
        dem_arrays = []
        for item in items:
            signed_href = item.assets["data"].href
            # Clip sesuai bbox agar ringan
            da = rioxarray.open_rasterio(signed_href).rio.clip_box(*bbox)
            dem_arrays.append(da)
            
        if len(dem_arrays) > 1:
            dem_merged = merge_arrays(dem_arrays)
            return dem_merged
        return dem_arrays[0]
    except Exception as e:
        st.error(f"Gagal Download DEM: {e}")
        return None

# ==============================================================================
# BAGIAN B: MESIN SIPIL (PCLP PARSER & DXF)
# ==============================================================================

def parse_pclp_block(df):
    """Parser format Excel PCLP Cross Section"""
    parsed_data = []
    i = 0
    df = df.astype(str) # Pastikan semua string dulu
    
    while i < len(df):
        row = df.iloc[i].values
        # Cari tanda 'X'
        x_indices = [idx for idx, val in enumerate(row) if str(val).strip().upper() == 'X']
        
        if x_indices and (i + 1 < len(df)):
            x_idx = x_indices[0]
            val_y = str(df.iloc[i+1, x_idx]).strip().upper()
            
            if val_y == 'Y': # Validasi blok PCLP
                # Ambil Nama STA
                sta_name = f"STA_{len(parsed_data)}"
                candidate = str(df.iloc[i+1, 1]).strip()
                if candidate.lower() not in ['nan', 'none', '']:
                    sta_name = candidate.replace('.0', '')

                # Ambil Data Titik
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
                    except: break # Stop jika ketemu non-angka
                
                if points:
                    points.sort(key=lambda p: p[0]) # Sort berdasarkan jarak X
                    parsed_data.append({'STA': sta_name, 'points': points})
                i += 1
        i += 1
    return parsed_data

def hitung_cut_fill(tanah, desain):
    """Hitung luas Cut & Fill sederhana dengan Shapely"""
    if not tanah or not desain: return 0.0, 0.0
    try:
        # Buat datum dummy di bawah tanah terendah
        min_y = min([p[1] for p in tanah] + [p[1] for p in desain])
        datum = min_y - 5.0
        
        # Buat Polygon tertutup
        p_tanah = tanah + [(tanah[-1][0], datum), (tanah[0][0], datum)]
        p_desain = desain + [(desain[-1][0], datum), (desain[0][0], datum)]
        
        poly_t = Polygon(p_tanah).buffer(0)
        poly_d = Polygon(p_desain).buffer(0)
        
        # Operasi Boolean
        area_cut = poly_d.intersection(poly_t).area
        area_fill = poly_d.difference(poly_t).area # Simplifikasi fill
        
        return area_cut, area_fill
    except: return 0.0, 0.0

def generate_dxf(results, mode="cross"):
    """Export hasil ke format DXF AutoCAD"""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Setup Layer
    doc.layers.add(name='TANAH', color=8)   # Abu-abu
    doc.layers.add(name='DESAIN', color=1)  # Merah
    doc.layers.add(name='TEXT', color=7)    # Putih

    if mode == "long":
        tanah, desain = results
        if tanah: msp.add_lwpolyline(tanah, dxfattribs={'layer': 'TANAH'})
        if desain: msp.add_lwpolyline(desain, dxfattribs={'layer': 'DESAIN'})
    else:
        # Mode Cross Section (Grid Layout)
        for i, item in enumerate(results):
            col = i % 2
            row = i // 2
            ox, oy = col * 60, row * -40 # Offset antar gambar
            
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
# BAGIAN C: ANTARMUKA UTAMA (SIDEBAR MENU)
# ==============================================================================

st.sidebar.title("🎛️ Menu Utama")
pilihan_menu = st.sidebar.radio("Pilih Modul:", ["1. Analisis Hidrologi (DAS)", "2. Desain Sipil (Cross/Long)"])

# ------------------------------------------------------------------------------
# HALAMAN 1: ANALISIS HIDROLOGI
# ------------------------------------------------------------------------------
if pilihan_menu == "1. Analisis Hidrologi (DAS)":
    st.title("💧 Analisis Hidrologi & Delineasi DAS")
    st.info("Upload batas wilayah (KML/GeoJSON) untuk mengunduh DEM dan analisis otomatis.")

    # 1. Upload Batas Wilayah
    up_file = st.file_uploader("Upload Batas Wilayah (GeoJSON/KML)", type=["geojson", "kml", "kmz"])
    
    if up_file:
        # Handling KML Driver
        try:
            fiona.drvsupport.supported_drivers['KML'] = 'rw'
            fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
        except: pass

        # Simpan file sementara untuk dibaca Geopandas
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_file.name.split('.')[-1]}") as tmp:
            tmp.write(up_file.getbuffer())
            tmp_path = tmp.name

        try:
            # Baca File
            gdf = gpd.read_file(tmp_path)
            # Konversi ke EPSG:4326 (WGS84) Wajib untuk Peta Web
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            bbox = gdf.total_bounds # [minx, miny, maxx, maxy]
            st.session_state['bbox'] = bbox
            st.success(f"Wilayah dimuat! Bound: {bbox}")
        except Exception as e:
            st.error(f"Gagal membaca file vektor: {e}")

    # 2. Download DEM
    if 'bbox' in st.session_state:
        if st.button("⬇️ Download & Proses DEM"):
            with st.spinner("Mengunduh DEM... (Mohon tunggu)"):
                dem_xr = fetch_dem_copernicus(st.session_state['bbox'])
                
                if dem_xr is not None:
                    # Simpan sebagai file fisik .tif (WAJIB untuk Pysheds)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f_dem:
                        dem_xr.rio.to_raster(f_dem.name)
                        st.session_state['dem_path'] = f_dem.name # Simpan path file
                    
                    st.success("DEM Berhasil Diunduh!")
                    
                    # Inisialisasi Hydro Engine
                    try:
                        eng = HydroEngine(st.session_state['dem_path'])
                        eng.condition_dem() # Proses Flow Direction
                        st.session_state['hydro_engine'] = eng
                        st.success("Analisis Arah Aliran Selesai!")
                    except Exception as e:
                        st.error(f"Error Hydro Engine: {e}")

    # 3. Peta Interaktif
    if 'dem_path' in st.session_state:
        st.subheader("Peta & Delineasi")
        st.caption("Klik pada area lembah/sungai di peta untuk membuat DAS.")
        
        m = leafmap.Map()
        m.add_raster(st.session_state['dem_path'], layer_name="DEM", colormap="terrain")
        
        # Tampilkan Peta
        map_out = st_folium(m, height=500, width=None)
        
        # Logika Klik Delineasi
        if map_out and map_out['last_clicked']:
            lat = map_out['last_clicked']['lat']
            lng = map_out['last_clicked']['lng']
            st.info(f"Koordinat Klik: {lat}, {lng}")
            
            if 'hydro_engine' in st.session_state:
                with st.spinner("Menghitung Batas DAS..."):
                    eng = st.session_state['hydro_engine']
                    poly_das = eng.delineate_catchment(lng, lat)
                    
                    if poly_das:
                        st.success(f"DAS Terbentuk! Luas: {poly_das.area:.6f} deg²")
                        
                        # Siapkan Download
                        gdf_res = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                        st.download_button("📥 Download GeoJSON DAS", gdf_res.to_json(), "das_result.geojson")
                    else:
                        st.warning("Gagal delineasi. Coba klik lebih pas di alur sungai.")

# ------------------------------------------------------------------------------
# HALAMAN 2: DESAIN SIPIL
# ------------------------------------------------------------------------------
elif pilihan_menu == "2. Desain Sipil (Cross/Long)":
    st.title("🚜 Desain Sipil & PCLP")
    
    tab1, tab2 = st.tabs(["📐 Cross Section", "📈 Long Section"])
    
    with tab1:
        st.subheader("Cross Section")
        f_pclp = st.file_uploader("Upload Excel PCLP (.xlsx)", type=['xlsx'])
        if f_pclp:
            xls = pd.ExcelFile(f_pclp)
            s_ogl = st.selectbox("Sheet Tanah", xls.sheet_names, index=0)
            s_dsn = st.selectbox("Sheet Desain", xls.sheet_names, index=1 if len(xls.sheet_names)>1 else 0)
            
            if st.button("Proses Data"):
                d_ogl = parse_pclp_block(pd.read_excel(f_pclp, sheet_name=s_ogl, header=None))
                d_dsn = parse_pclp_block(pd.read_excel(f_pclp, sheet_name=s_dsn, header=None))
                
                # Gabung
                final = []
                for i in range(max(len(d_ogl), len(d_dsn))):
                    t = d_ogl[i] if i < len(d_ogl) else None
                    d = d_dsn[i] if i < len(d_dsn) else None
                    sta = t['STA'] if t else (d['STA'] if d else f"STA_{i}")
                    tp, dp = (t['points'] if t else []), (d['points'] if d else [])
                    c, f = hitung_cut_fill(tp, dp)
                    final.append({'STA': sta, 'points_tanah': tp, 'points_desain': dp, 'cut': c, 'fill': f})
                
                st.session_state['cross_res'] = final
                st.success("Selesai!")
        
        if 'cross_res' in st.session_state:
            res = st.session_state['cross_res']
            idx = st.slider("Pilih STA", 0, len(res)-1, 0)
            item = res[idx]
            
            fig, ax = plt.subplots(figsize=(10,3))
            if item['points_tanah']: ax.plot(*zip(*item['points_tanah']), 'k-o', label='Tanah')
            if item['points_desain']: ax.plot(*zip(*item['points_desain']), 'r-', label='Desain')
            ax.set_title(f"{item['STA']} | C: {item['cut']:.2f}, F: {item['fill']:.2f}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            dxf_btn = generate_dxf(res, "cross")
            st.download_button("📥 Download DXF", dxf_btn, "cross_section.dxf")
            
    with tab2:
        st.subheader("Long Section")
        f_long = st.file_uploader("Upload Long Section (.csv)", type=['csv'])
        if f_long:
            df = pd.read_csv(f_long)
            # Asumsi 2 kolom pertama
            pts = df.iloc[:, :2].dropna().values.tolist()
            st.session_state['long_pts'] = pts
            
        if 'long_pts' in st.session_state:
            pts = st.session_state['long_pts']
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(*zip(*pts), 'b-')
            st.pyplot(fig)
            
            dxf_long = generate_dxf((pts, []), "long")
            st.download_button("📥 Download DXF Long", dxf_long, "long_section.dxf")
