import streamlit as st
import tempfile
import os
import shutil

# Import Library
try:
    import geopandas as gpd
    import rasterio
    from pysheds.grid import Grid
    import leafmap.foliumap as leafmap
    from streamlit_folium import st_folium
    from shapely.geometry import Polygon
    import fiona
except: pass

# --- HYDRO ENGINE ---
class HydroEngine:
    def __init__(self, dem_path):
        self.grid = Grid.from_raster(dem_path)
        self.dem = self.grid.read_raster(dem_path)
        self.grid.add_gridded_data(self.dem, data_name='dem')
        self.fdir = None 
        self.acc = None

    def condition_dem(self):
        flooded_dem = self.grid.fill_depressions(self.dem)
        inflated_dem = self.grid.resolve_flats(flooded_dem)
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        self.fdir = self.grid.flowdir(inflated_dem, dirmap=dirmap)
        self.acc = self.grid.accumulation(self.fdir, dirmap=dirmap)

    def delineate(self, x, y):
        if self.fdir is None: return None
        try:
            snapped = self.grid.snap_to_mask(self.acc > 100, (x, y))
            catch = self.grid.catchment(x=snapped[0], y=snapped[1], fdir=self.fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32), xytype='coordinate')
            self.grid.clip_to(catch)
            shapes = self.grid.polygonize()
            for shape, value in shapes:
                if value == 1: return Polygon(shape['coordinates'][0])
        except: return None

# --- UI ---
st.title("💧 Analisis Hidrologi & DAS")
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("1. Sumber Data")
    
    # --- LOGIKA AMBIL DATA (FIXED) ---
    # Cek apakah ada PATH file yang dishare dari sebelah
    shared_path = st.session_state.get('shared_dem_path')
    
    final_dem_path = None
    
    # KONDISI 1: Ada data dari sebelah
    if shared_path and os.path.exists(shared_path):
        st.info("📦 Terdeteksi DEM dari Modul Sipil")
        if st.button("🔄 Pakai Data Modul Sipil"):
            st.session_state['active_dem'] = shared_path
            st.success("Data dimuat!")
            st.rerun()
            
    # KONDISI 2: Upload Manual (Jika tidak pakai data sebelah)
    if 'active_dem' not in st.session_state:
        up = st.file_uploader("Atau Upload Manual (.tif)", type=['tif'])
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f:
                f.write(up.getbuffer())
                st.session_state['active_dem'] = f.name
                st.rerun()

    # Tombol Reset
    if 'active_dem' in st.session_state:
        if st.button("❌ Ganti File"):
            del st.session_state['active_dem']
            st.rerun()
            
    st.markdown("---")
    btn_proc = st.button("🌊 PROSES FLOW DIRECTION")

with c2:
    if btn_proc and 'active_dem' in st.session_state:
        with st.spinner("Menganalisis Arah Aliran..."):
            try:
                eng = HydroEngine(st.session_state['active_dem'])
                eng.condition_dem()
                st.session_state['engine'] = eng
                st.success("✅ Selesai! Klik peta untuk delineasi.")
            except Exception as e: st.error(f"Error: {e}")

    if 'active_dem' in st.session_state:
        m = leafmap.Map()
        try:
            m.add_raster(st.session_state['active_dem'], layer_name="DEM", colormap="terrain")
        except: pass
        
        out = st_folium(m, height=500)
        
        if out and out['last_clicked'] and 'engine' in st.session_state:
            lat, lng = out['last_clicked']['lat'], out['last_clicked']['lng']
            st.info(f"Titik: {lat}, {lng}")
            with st.spinner("Delineasi..."):
                poly = st.session_state['engine'].delineate(lng, lat)
                if poly:
                    st.success(f"Luas DAS: {(poly.area * 12391):.2f} km² (Estimasi)")
                    gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326").to_file("das.geojson", driver="GeoJSON")
                    with open("das.geojson", "rb") as f:
                        st.download_button("Download GeoJSON", f, "das.geojson")
                else: st.warning("Coba klik di sungai.")
