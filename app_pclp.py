import streamlit as st
import io  # [FIX] Dipindahkan ke paling atas
import os
import tempfile
import zipfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium  # [FIX] Untuk interaksi peta
from pystac_client import Client
import planetary_computer
import rioxarray
from rioxarray.merge import merge_arrays
from pysheds.grid import Grid
from shapely.geometry import shape, box
import fiona

# ==========================================
# 0. KONFIGURASI HALAMAN & DRIVER
# ==========================================
st.set_page_config(page_title="HydroStream: Advanced Hydrology", layout="wide", page_icon="💧")

# [CRITICAL] Enable KML Driver support
# Menggunakan try-except agar tidak error jika driver sudah aktif
try:
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
except Exception:
    pass

# ==========================================
# 1. FUNGSI LOAD DATA (UNIFIED LOADER)
# ==========================================
def load_vector_data(uploaded_file):
    """
    Menangani input hybrid: GeoJSON (teks) atau KMZ (zip binary).
    """
    try:
        # Buat file sementara agar fiona bisa membacanya dengan stabil
        suffix = ".kmz" if uploaded_file.name.lower().endswith(".kmz") else ".json"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        gdf = None
        
        # Skenario 1: KMZ (Harus di-unzip dulu)
        if uploaded_file.name.lower().endswith('.kmz'):
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(tmp_path, 'r') as z:
                    z.extractall(temp_dir)
                    kml_files = [x for x in os.listdir(temp_dir) if x.endswith(".kml")]
                    
                    if kml_files:
                        kml_path = os.path.join(temp_dir, kml_files[0])
                        gdf = gpd.read_file(kml_path, driver='KML')
                    else:
                        st.error("File KMZ valid, tapi tidak ada .kml di dalamnya.")
        
        # Skenario 2: GeoJSON
        else:
            gdf = gpd.read_file(tmp_path)

        # Bersihkan file sementara
        os.remove(tmp_path)

        # Standardisasi ke WGS84
        if gdf is not None:
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
            
    except Exception as e:
        st.error(f"Error membaca file: {str(e)}")
        return None

# ==========================================
# 2. ACQUISITION DEM (STAC API)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_dem_copernicus(bounds):
    """Mengunduh DEM dari Planetary Computer."""
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace
        )
        
        bbox = [bounds[0]-0.01, bounds[1]-0.01, bounds[2]+0.01, bounds[3]+0.01]
        search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
        items = list(search.get_items())
        
        if not items:
            return None, "Tidak ada data DEM di lokasi ini."

        datasets = []
        for item in items:
            href = item.assets["data"].href
            da = rioxarray.open_rasterio(href).rio.clip_box(*bbox)
            datasets.append(da)

        if len(datasets) > 1:
            dem_merged = merge_arrays(datasets)
        else:
            dem_merged = datasets[0]

        # Simpan ke BytesIO agar bisa dibaca Pysheds
        dem_bytes = io.BytesIO()
        dem_merged.rio.to_raster(dem_bytes, driver="GTiff")
        dem_bytes.seek(0)
        return dem_bytes, None

    except Exception as e:
        return None, str(e)

# ==========================================
# 3. HYDROLOGY ENGINE
# ==========================================
class HydroEngine:
    def __init__(self, dem_bytes):
        self.grid = Grid()
        self.grid.read_raster(dem_bytes)
        self.dem = self.grid.view('dem')
        
    def condition_dem(self):
        # Mengisi cekungan (Pit Filling)
        self.grid.fill_depressions('dem', out_name='flooded_dem')
        # Menyelesaikan area datar (Resolve Flats)
        self.grid.resolve_flats('flooded_dem', out_name='inflated_dem')
        return self.grid.view('inflated_dem')

    def compute_flow(self):
        # Arah Aliran (D8)
        self.grid.flowdir(data='inflated_dem', out_name='fdir', dirmap=(64, 128, 1, 2, 4, 8, 16, 32))
        # Akumulasi Aliran
        self.grid.accumulation(data='fdir', out_name='acc', dirmap=(64, 128, 1, 2, 4, 8, 16, 32))
        return self.grid.view('acc')

    def get_stream_network(self, threshold=1000):
        acc = self.grid.view('acc')
        return acc > threshold

# ==========================================
# 4. MAIN UI
# ==========================================
def main():
    st.sidebar.title("🛠️ Kontrol Hidrologi")
    
    # --- INPUT ---
    uploaded_file = st.sidebar.file_uploader("Upload AOI (GeoJSON/KMZ)", type=['geojson', 'kml', 'kmz'])
    aoi_gdf = None
    
    if uploaded_file:
        aoi_gdf = load_vector_data(uploaded_file)
        if aoi_gdf is not None:
            st.sidebar.success(f"AOI: {len(aoi_gdf)} fitur.")

    # --- INITIALIZE MAP ---
    m = leafmap.Map(draw_control=False, google_map="HYBRID")
    
    # --- PROCESSING ---
    if aoi_gdf is not None:
        m.add_gdf(aoi_gdf, layer_name="Batas AOI", style={'color': 'red', 'fill': False})
        bounds = aoi_gdf.total_bounds
        m.zoom_to_bounds(bounds)
        
        if st.sidebar.button("🚀 Proses Hidrologi"):
            with st.spinner("Mengunduh DEM & Analisis..."):
                dem_bytes, err = fetch_dem_copernicus(bounds)
                if dem_bytes:
                    eng = HydroEngine(dem_bytes)
                    eng.condition_dem()
                    eng.compute_flow()
                    st.session_state['engine'] = eng
                    st.session_state['dem_ok'] = True
                else:
                    st.error(err)

    # --- VISUALIZATION ---
    if st.session_state.get('dem_ok'):
        engine = st.session_state['engine']
        
        # 1. Slider Sungai
        thresh = st.sidebar.slider("Threshold Sungai", 100, 5000, 1000)
        streams = engine.get_stream_network(thresh)
        stream_view = np.where(streams, 1, np.nan)
        m.add_raster(engine.grid.to_raster(stream_view, data=stream_view), 
                     layer_name="Sungai", palette="Blues", vmin=0, vmax=1)
        
        # 2. Slider Banjir
        flood_h = st.sidebar.slider("Tinggi Banjir (m)", 0.0, 20.0, 0.0)
        if flood_h > 0:
            dem = engine.dem
            min_h = np.nanmin(dem)
            mask = dem < (min_h + flood_h)
            view = np.where(mask, 1, np.nan)
            m.add_raster(engine.grid.to_raster(view, data=view), 
                         layer_name="Banjir", palette="Reds", opacity=0.5)

    # --- RENDER MAP ---
    # Gunakan st_folium untuk menangkap interaksi, bukan m.to_streamlit()
    st_data = st_folium(m, height=600, width=None)

if __name__ == "__main__":
    main()
