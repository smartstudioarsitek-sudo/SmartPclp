import streamlit as st
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
import geopandas as gpd
import fiona
import tempfile
import os
from pysheds.grid import Grid
from shapely.geometry import Polygon
import numpy as np

# --- 1. DEFINISI CLASS ENGINE (OTAK PERHITUNGAN) ---
class HydroEngine:
    def __init__(self, dem_path):
        # 1. Baca Grid dari File DEM
        self.grid = Grid.from_raster(dem_path)
        self.dem = self.grid.read_raster(dem_path)
        
        # 2. Pre-processing Hidrologi (Otomatis saat upload)
        # Menentukan arah aliran (N, NE, E, SE, S, SW, W, NW)
        self.dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        
        # Fill Depressions (Mengisi cekungan agar air mengalir)
        self.pit_filled = self.grid.fill_depressions(self.dem)
        self.flooded = self.grid.resolve_flats(self.pit_filled)
        
        # Flow Direction (Arah Aliran)
        self.fdir = self.grid.flowdir(self.flooded, dirmap=self.dirmap)
        
        # Flow Accumulation (Akumulasi Aliran)
        self.acc = self.grid.accumulation(self.fdir, dirmap=self.dirmap)

    def delineate(self, x, y):
        # 3. Fungsi Delineasi berdasarkan titik klik
        try:
            # Snap titik klik ke aliran sungai terdekat (akumulasi tinggi)
            # Mencari titik dengan akumulasi aliran > 100 cell di sekitar klik
            xy = self.grid.snap_to_mask(self.acc > 100, (x, y))
            
            # Hitung Catchment Area (DAS)
            catch = self.grid.catchment(x=xy[0], y=xy[1], fdir=self.fdir, dirmap=self.dirmap, xytype='coordinate')
            
            # Grid catchment dipotong agar sesuai bounding box
            self.grid.clip_to(catch)
            clipped_catch = self.grid.view(catch)
            
            # Konversi Grid Raster ke Vector (Polygon)
            shapes = self.grid.polygonize(clipped_catch)
            
            # Ambil polygon hasil (biasanya value > 0)
            for shape, value in shapes:
                if value > 0:
                    # Kembalikan sebagai object Shapely Polygon
                    return Polygon(shape['coordinates'][0])
            return None
        except Exception as e:
            st.error(f"Gagal delineasi: {e}")
            return None

# --- 2. JUDUL & SETUP HALAMAN ---
st.title("💧 Analisis Hidrologi & Delineasi DAS")

# Inisialisasi Session State
if 'active_dem' not in st.session_state:
    st.session_state['active_dem'] = None
if 'engine' not in st.session_state:
    st.session_state['engine'] = None

# --- 3. INPUT DATA ---
st.subheader("1. Input Data Topografi")
uploaded_dem = st.file_uploader("Upload File DEM (.tif)", type=["tif", "tiff"])
up_boundary = st.file_uploader("Upload Batas Wilayah (Opsional - .kml/.geojson)", type=["kml", "json", "geojson"])

# LOGIKA PROSES FILE DEM
if uploaded_dem is not None:
    # Cek apakah file baru berbeda dengan yang ada di memory
    # Agar tidak memproses ulang jika file sama
    is_new_file = True
    if st.session_state['active_dem'] is not None:
        if uploaded_dem.name in st.session_state['active_dem']:
            is_new_file = False

    if is_new_file:
        try:
            # Simpan file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(uploaded_dem.getbuffer())
                dem_path = tmp.name
            
            st.session_state['active_dem'] = dem_path
            
            # --- INISIALISASI ENGINE (INI YANG TADI HILANG) ---
            with st.spinner("⚙️ Sedang memproses data hidrologi (Fill Pits & Flow Direction)... Mohon tunggu."):
                # Membuat object HydroEngine baru
                eng = HydroEngine(dem_path)
                st.session_state['engine'] = eng
            
            st.success(f"✅ Analisis Hidrologi Selesai! Silakan klik peta.")
            
        except Exception as e:
            st.error(f"Gagal memuat file DEM: {e}")

# --- 4. PETA INTERAKTIF ---
st.divider()

if st.session_state['active_dem']:
    st.subheader("2. Peta Interaktif")
    
    m = leafmap.Map(google_map="HYBRID") 
    
    # Layer DEM
    try:
        m.add_raster(st.session_state['active_dem'], layer_name="Topografi (DEM)", colormap="terrain", opacity=0.6)
    except Exception as e:
        st.caption(f"ℹ️ Visualisasi raster skip: {e}")

    # Layer Batas Wilayah
    try:
        if up_boundary is not None:
            fiona.drvsupport.supported_drivers['KML'] = 'rw'
            fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_boundary.name.split('.')[-1]}") as tmp:
                tmp.write(up_boundary.getbuffer())
                tmp_path = tmp.name
            gdf = gpd.read_file(tmp_path)
            bounds = gdf.total_bounds
            m.zoom_to_bounds((bounds[0], bounds[1], bounds[2], bounds[3]))
            style = {'fillColor': '#00000000', 'color': 'cyan', 'weight': 3}
            m.add_gdf(gdf, layer_name="Batas Wilayah", style=style)
    except: pass
        
    map_out = st_folium(m, height=500, width=None)
    
    # --- 5. LOGIKA KLIK & HASIL ---
    if map_out and map_out['last_clicked']:
        lat = map_out['last_clicked']['lat']
        lng = map_out['last_clicked']['lng']
        st.info(f"📍 Koordinat Klik: {lat:.5f}, {lng:.5f}")
        
        # Cek Engine
        if st.session_state['engine'] is not None:
            with st.spinner("⏳ Sedang menelusuri batas DAS..."):
                eng = st.session_state['engine']
                
                # JALANKAN DELINEASI
                poly_das = eng.delineate(lng, lat)
                
                if poly_das:
                    # Hitung Luas (Estimasi kasar)
                    area_km2 = (poly_das.area * 111.32 * 111.32) 
                    st.success(f"✅ DAS Berhasil Dibuat! Luas: ±{area_km2:.2f} km²")
                    
                    # Buat GeoJSON untuk Download
                    gdf_res = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                    
                    # Kolom Download
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download GeoJSON",
                            data=gdf_res.to_json(),
                            file_name="das_result.geojson",
                            mime="application/json"
                        )
                    
                    # Tampilkan Preview DAS di Peta Statis (Matplotlib) untuk konfirmasi
                    # st.pyplot(...) # Opsional jika ingin preview grafik
                else:
                    st.warning("⚠️ Titik klik tidak valid (di luar aliran sungai). Coba klik di area lembah.")
        else:
            st.error("⚠️ Engine belum siap. Silakan upload ulang DEM.")
else:
    st.info("👈 Silakan upload file DEM terlebih dahulu.")
