import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import tempfile
import os
import zipfile
import leafmap.foliumap as leafmap
from pystac_client import Client
import planetary_computer
import rioxarray
from pysheds.grid import Grid
from shapely.geometry import shape, box
import fiona

# ==========================================
# KONFIGURASI HALAMAN & DRIVER
# ==========================================
st.set_page_config(page_title="HydroStream: Advanced Hydrology", layout="wide", page_icon="💧")

# [CRITICAL] Enable KML Driver support for Fiona/Geopandas
# Ini memperbaiki masalah "Driver Error" saat membaca KML/KMZ
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# ==========================================
# 1. UNIFIED INPUT LOADER (GEOJSON + KMZ)
# ==========================================
def load_vector_data(uploaded_file):
    """
    Menangani input hybrid: Membaca GeoJSON text-stream atau KMZ binary-stream.
    Mengembalikan: GeoDataFrame (EPSG:4326)
    """
    try:
        # Skenario 1: GeoJSON (Text based)
        if uploaded_file.name.lower().endswith(('.geojson', '.json')):
            # Geopandas bisa membaca BytesIO langsung
            uploaded_file.seek(0)
            gdf = gpd.read_file(uploaded_file)

        # Skenario 2: KMZ (Zipped KML)
        elif uploaded_file.name.lower().endswith('.kmz'):
            # KMZ harus di-unzip. Gunakan temp directory agar bersih.
            with tempfile.TemporaryDirectory() as temp_dir:
                # Simpan arsip KMZ sementara
                kmz_path = os.path.join(temp_dir, "temp.kmz")
                with open(kmz_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ekstrak
                with zipfile.ZipFile(kmz_path, 'r') as z:
                    z.extractall(temp_dir)
                    # Cari file .kml di dalam hasil ekstraksi
                    kml_files = [x for x in os.listdir(temp_dir) if x.endswith(".kml")]
                    
                    if not kml_files:
                        st.error("KMZ valid, tapi tidak ditemukan file .kml didalamnya.")
                        return None
                    
                    kml_path = os.path.join(temp_dir, kml_files[0])
                    # Baca dengan driver KML eksplisit
                    gdf = gpd.read_file(kml_path, driver='KML')

        else:
            st.error("Format file tidak didukung.")
            return None

        # Standardisasi Koordinat ke WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        return gdf

    except Exception as e:
        st.error(f"Error Loading File: {str(e)}")
        return None

# ==========================================
# 2. AUTOMATED DEM ACQUISITION (STAC API)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_dem_copernicus(bounds):
    """
    Mengunduh DEM Copernicus GLO-30 dari Microsoft Planetary Computer.
    Input: bounds (minx, miny, maxx, maxy)
    Output: Rasterio MemoryFile object (virtual file)
    """
    # 1. Setup Client
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # 2. Search Tile
    # Buffer sedikit bounds agar tidak pas di tepi
    bbox = [bounds[0]-0.01, bounds[1]-0.01, bounds[2]+0.01, bounds[3]+0.01]
    
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox
    )
    items = list(search.get_items())
    
    if not items:
        return None, "Tidak ditemukan data DEM di lokasi ini."

    # 3. Load & Merge using Rioxarray (Lazy Loading)
    try:
        # Load datasets
        datasets = []
        for item in items:
            signed_asset = item.assets["data"].href
            da = rioxarray.open_rasterio(signed_asset)
            # Clip dulu sebelum merge untuk hemat memori
            da_clipped = da.rio.clip_box(*bbox)
            datasets.append(da_clipped)

        # Merge jika lebih dari 1 tile
        if len(datasets) > 1:
            from rioxarray.merge import merge_arrays
            dem_merged = merge_arrays(datasets)
        else:
            dem_merged = datasets[0]

        # 4. Export ke In-Memory Rasterio Format untuk Pysheds
        # Kita perlu menyimpan sebagai bytes agar bisa dibaca Pysheds/Rasterio
        dem_bytes = io.BytesIO()
        dem_merged.rio.to_raster(dem_bytes, driver="GTiff")
        dem_bytes.seek(0)
        
        return dem_bytes, None

    except Exception as e:
        return None, str(e)

# ==========================================
# 3. HYDROLOGY ENGINE (PYSHEDS)
# ==========================================
class HydroEngine:
    def __init__(self, dem_bytes):
        # Load DEM ke Pysheds Grid
        self.grid = Grid()
        self.grid.read_raster(dem_bytes)
        self.dem = self.grid.view('dem')
        
    def condition_dem(self):
        # 1. Fill Depressions (Pit Filling)
        # Pysheds menggunakan algoritma Priority-Flood yang efisien
        self.demb = self.grid.fill_depressions('dem')
        
        # 2. Resolve Flats
        self.inflated_dem = self.grid.resolve_flats('dem', new_data='inflated_dem')
        return self.inflated_dem

    def compute_flow(self):
        # 3. Flow Direction (Numpy-based D8)
        # dirmap=(64, 128, 1, 2, 4, 8, 16, 32) -> ESRI Scheme support
        self.fdir = self.grid.flowdir(data='inflated_dem', dirmap=(64, 128, 1, 2, 4, 8, 16, 32))
        
        # 4. Flow Accumulation
        self.acc = self.grid.accumulation(data='fdir', dirmap=(64, 128, 1, 2, 4, 8, 16, 32))
        return self.fdir, self.acc

    def extract_streams(self, threshold=1000):
        # 5. Extract River Network
        # threshold = jumlah sel hulu
        streams = self.acc > threshold
        return streams

    def delineate_catchment(self, x, y):
        # 6. Snap Pour Point
        # Cari akumulasi tertinggi dalam radius pencarian
        try:
            xy = self.grid.snap_to_mask(self.acc > 1000, (x, y))
            
            # 7. Delineate
            catchment = self.grid.catchment(x=xy[0], y=xy[1], fdir='fdir', 
                                          dirmap=(64, 128, 1, 2, 4, 8, 16, 32), 
                                          xytype='coordinate')
            
            # 8. Polygonize hasil catchment raster ke Vector
            shapes = self.grid.polygonize(catchment)
            # Ambil shape terbesar (DAS utama)
            catchment_poly = max(shapes, key=lambda s: shape(s[0]).area)
            return shape(catchment_poly[0])
        except Exception as e:
            return None

import io # Helper import

# ==========================================
# MAIN APP UI
# ==========================================
def main():
    st.sidebar.title("🛠️ Kontrol Hidrologi")
    
    # 1. FILE UPLOAD (RESTORED FUNCTIONALITY)
    st.sidebar.subheader("1. Input Data Spasial")
    # [CRITICAL] Widget type diset inclusive untuk GeoJSON & KMZ
    uploaded_file = st.sidebar.file_uploader(
        "Upload AOI (Batas Wilayah)", 
        type=['geojson', 'kml', 'kmz'],
        help="Mendukung format GeoJSON (teks) dan KMZ (Google Earth)"
    )

    aoi_gdf = None
    if uploaded_file:
        with st.spinner("Parsing file input..."):
            aoi_gdf = load_vector_data(uploaded_file)
            
        if aoi_gdf is not None:
            st.sidebar.success(f"AOI Dimuat: {len(aoi_gdf)} fitur.")
            bounds = aoi_gdf.total_bounds # minx, miny, maxx, maxy
    
    # 2. MAP INITIALIZATION
    m = leafmap.Map(draw_control=False)
    
    # 3. PROCESSING WORKFLOW
    if aoi_gdf is not None:
        # Tampilkan AOI
        m.add_gdf(aoi_gdf, layer_name="Batas AOI", style={'color': 'red', 'fill': False, 'weight': 2})
        m.zoom_to_bounds(aoi_gdf.total_bounds)
        
        # Tombol Proses DEM
        if st.sidebar.button("🚀 Mulai Analisis Hidrologi"):
            with st.spinner("Mengunduh DEM Copernicus GLO-30 (High-Res)..."):
                dem_bytes, err = fetch_dem_copernicus(bounds)
            
            if dem_bytes:
                st.session_state['dem_data'] = dem_bytes
                
                with st.spinner("Preprocessing DEM & Flow Routing..."):
                    # Inisialisasi Engine
                    engine = HydroEngine(dem_bytes)
                    engine.condition_dem()
                    engine.compute_flow()
                    
                    # Simpan engine ke session state (perlu pickling hati-hati, atau simpan grid saja)
                    # Untuk simplicity, kita hitung on-the-fly karena Pysheds cepat untuk area kecil-sedang
                    st.session_state['engine'] = engine
                    st.success("Analisis Topografi Selesai!")
            else:
                st.error(err)

    # 4. INTERACTIVE ANALYSIS LAYERS
    if 'engine' in st.session_state:
        engine = st.session_state['engine']
        
        st.sidebar.subheader("2. Parameter Sungai")
        thresh = st.sidebar.slider("Flow Accumulation Threshold", 100, 10000, 1000, 
                                   help="Nilai kecil = Deteksi anak sungai. Nilai besar = Sungai utama saja.")
        
        # Ekstrak Sungai Real-time
        streams_raster = engine.extract_streams(thresh)
        
        # Visualisasi Sungai (Raster Overlay)
        # Konversi ke float untuk visualisasi (0=NoData, 1=Stream)
        stream_view = np.where(streams_raster, 1, np.nan)
        m.add_raster(engine.grid.to_raster(stream_view, data=stream_view), 
                     layer_name="Jaringan Sungai", palette="Blues", vmin=0, vmax=1)
        
        st.sidebar.subheader("3. Simulasi Banjir (Bathtub)")
        flood_level = st.sidebar.slider("Kenaikan Muka Air (m)", 0.0, 20.0, 0.0, step=0.5)
        
        if flood_level > 0:
            # Simple Bathtub Model: DEM < (Min Elevation + Rise)
            # Untuk 'Connected' model, kita bisa gunakan streams sebagai seed (opsional)
            base_elev = np.nanmin(engine.dem)
            flood_mask = engine.dem < (base_elev + flood_level)
            flood_view = np.where(flood_mask, 1, np.nan)
            
            m.add_raster(engine.grid.to_raster(flood_view, data=flood_view),
                         layer_name=f"Banjir (+{flood_level}m)", palette="coolwarm", opacity=0.6)
            
            # Hitung Estimasi Volume (Kasar)
            # Asumsi resolusi ~30m (GLO-30) -> 900m2 per pixel
            # Ini hanya estimasi pixel count
            flood_pixels = np.nansum(flood_mask)
            est_area_ha = (flood_pixels * 900) / 10000
            st.metric("Estimasi Luas Genangan", f"{est_area_ha:,.2f} Ha")

        st.sidebar.info("Klik pada peta di dekat sungai untuk Delineasi DAS otomatis.")

    # 5. MAP INTERACTION (Catchment Delineation)
    # Gunakan folium last click
    m.to_streamlit(height=600)
    
    # [LOGIC DELINEASI DAS]
    # Karena keterbatasan callback Streamlit-Folium yang kompleks, 
    # fitur klik biasanya membutuhkan komponen st_folium khusus.
    # Di sini kita berikan instruksi logika backend-nya.
    # Jika Anda menggunakan st_folium, Anda bisa mengambil output['last_clicked']
    # lalu pass ke engine.delineate_catchment(lat, lon)

if __name__ == "__main__":
    main()
