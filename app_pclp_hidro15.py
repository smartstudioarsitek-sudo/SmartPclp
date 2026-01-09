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

# --- 1. DEFINISI CLASS ENGINE DENGAN CACHING ---
# Kita gunakan decorator @st.cache_resource agar class ini HANYA dimuat 1x saja
# sampai file inputnya berubah. Ini menghemat memori dan waktu loading drastis.
@st.cache_resource(show_spinner="Sedang memproses topografi (Flow Direction)...")
def get_hydro_engine(dem_path):
    # Class didefinisikan di dalam fungsi cache atau dipanggil helper
    class HydroEngine:
        def __init__(self, path):
            self.grid = Grid.from_raster(path)
            self.dem = self.grid.read_raster(path)
            # Pre-processing
            self.dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
            self.pit_filled = self.grid.fill_depressions(self.dem)
            self.flooded = self.grid.resolve_flats(self.pit_filled)
            self.fdir = self.grid.flowdir(self.flooded, dirmap=self.dirmap)
            self.acc = self.grid.accumulation(self.fdir, dirmap=self.dirmap)

        def delineate(self, x, y):
            try:
                # Snap titik ke sungai terdekat (threshold 100 cell)
                xy = self.grid.snap_to_mask(self.acc > 100, (x, y))
                catch = self.grid.catchment(x=xy[0], y=xy[1], fdir=self.fdir, dirmap=self.dirmap, xytype='coordinate')
                self.grid.clip_to(catch)
                clipped_catch = self.grid.view(catch)
                shapes = self.grid.polygonize(clipped_catch)
                for shape, value in shapes:
                    if value > 0:
                        return Polygon(shape['coordinates'][0])
                return None
            except Exception as e:
                return None
    
    # Return object engine yang sudah jadi
    return HydroEngine(dem_path)

# --- 2. SETUP HALAMAN ---
st.set_page_config(layout="wide", page_title="SmartPCLP Hidro")
st.title("💧 Analisis Hidrologi & Delineasi DAS")

# Inisialisasi Session State untuk Koordinat
if 'selected_lat' not in st.session_state:
    st.session_state['selected_lat'] = 0.0
if 'selected_lng' not in st.session_state:
    st.session_state['selected_lng'] = 0.0

# --- 3. INPUT DATA ---
with st.sidebar:
    st.header("1. Input Data")
    uploaded_dem = st.file_uploader("Upload DEM (.tif)", type=["tif"])
    up_boundary = st.file_uploader("Upload Batas Wilayah (Opsional)", type=["kml", "json", "geojson"])

# --- 4. LOGIKA UTAMA ---
if uploaded_dem is not None:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_dem.getbuffer())
        dem_path = tmp.name
    
    # Memuat Engine (Akan cepat karena dicache)
    try:
        eng = get_hydro_engine(dem_path)
        st.sidebar.success(f"✅ Engine Siap!")
    except Exception as e:
        st.error(f"Error memuat DEM: {e}")
        st.stop()

    # --- LAYOUT DUA KOLOM (PETA & KONTROL) ---
    col_map, col_control = st.columns([3, 1])

    with col_map:
        st.subheader("2. Peta Interaktif")
        m = leafmap.Map(google_map="HYBRID")
        
        # Tambah Layer DEM
        m.add_raster(dem_path, layer_name="Topografi", colormap="terrain", opacity=0.6)
        
        # Tambah Boundary jika ada
        if up_boundary:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_boundary.name.split('.')[-1]}") as tmp:
                    tmp.write(up_boundary.getbuffer())
                    tmp_path = tmp.name
                gdf = gpd.read_file(tmp_path)
                style = {'fillColor': '#00000000', 'color': 'cyan', 'weight': 3}
                m.add_gdf(gdf, layer_name="Batas Wilayah", style=style)
                # Zoom ke boundary
                bounds = gdf.total_bounds
                m.zoom_to_bounds((bounds[0], bounds[1], bounds[2], bounds[3]))
            except: pass

        # Tampilkan Peta
        # PENTING: Kita ambil return value map_out untuk mendeteksi klik
        map_out = st_folium(m, height=600, width=None)

    # --- LOGIKA UPDATE KOORDINAT ---
    # Jika user klik peta, update session state
    if map_out and map_out['last_clicked']:
        st.session_state['selected_lat'] = map_out['last_clicked']['lat']
        st.session_state['selected_lng'] = map_out['last_clicked']['lng']

    with col_control:
        st.subheader("3. Kontrol Proses")
        st.write("Klik di peta ATAU isi koordinat manual di bawah ini:")
        
        # Form Input Manual (Terisi otomatis jika klik peta)
        # Kita gunakan st.form agar tidak reload tiap ketik angka
        with st.form("form_delineasi"):
            input_lat = st.number_input("Latitude", value=st.session_state['selected_lat'], format="%.6f")
            input_lng = st.number_input("Longitude", value=st.session_state['selected_lng'], format="%.6f")
            
            st.info("Tips: Klik area lembah/sungai, bukan puncak bukit.")
            
            # --- TOMBOL RUN ---
            # Proses hanya jalan kalau tombol ini ditekan!
            submit_btn = st.form_submit_button("🚀 HITUNG DAS SEKARANG", type="primary")

        # --- EKSEKUSI PROSES ---
        if submit_btn:
            with st.spinner("⏳ Sedang menghitung catchment area..."):
                # Gunakan koordinat dari input box (bisa hasil klik atau ketikan manual)
                poly_das = eng.delineate(input_lng, input_lat)
                
                if poly_das:
                    area_km2 = (poly_das.area * 111.32 * 111.32)
                    st.success(f"✅ Sukses! Luas: {area_km2:.2f} km²")
                    
                    # Download GeoJSON
                    gdf_res = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                    st.download_button(
                        "📥 Download GeoJSON",
                        data=gdf_res.to_json(),
                        file_name="das_result.geojson",
                        mime="application/json"
                    )
                    
                    # Tampilkan polygon hasil di peta statis kecil sebagai konfirmasi (opsional)
                    st.write("Preview Bentuk DAS:")
                    st.json(poly_das.__geo_interface__) # Atau visualisasi matplotlib simple
                else:
                    st.error("⚠️ Gagal delineasi. Titik mungkin diluar aliran sungai.")

else:
    st.info("👈 Upload file DEM di sidebar untuk memulai.")