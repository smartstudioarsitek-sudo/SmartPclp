import streamlit as st
import tempfile
import os

# --- IMPORT LIBRARY ---
try:
    import geopandas as gpd
    import rasterio
    from pysheds.grid import Grid
    import leafmap.foliumap as leafmap
    from streamlit_folium import st_folium
    from shapely.geometry import Polygon
    import fiona
except ImportError:
    pass

# ==========================================
# HYDRO ENGINE
# ==========================================
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
        return None

# ==========================================
# UI HIDROLOGI
# ==========================================
st.title("💧 Analisis Hidrologi & DAS")
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("1. Sumber Data")
    
    # Cek Data Kiriman dari Modul Sipil
    shared_path = st.session_state.get('shared_dem_path')
    
    # Tampilkan Tombol Ambil Data jika ada
    if shared_path and os.path.exists(shared_path):
        st.info("📦 Terdeteksi DEM dari Modul Sipil")
        if st.button("🔄 Pakai Data Modul Sipil"):
            st.session_state['active_dem'] = shared_path
            st.success("Data berhasil dimuat!")
            st.rerun()
            
    # Tampilkan Upload Manual jika belum ada data aktif
    if 'active_dem' not in st.session_state:
        up = st.file_uploader("Atau Upload Manual (.tif)", type=['tif'])
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f:
                f.write(up.getbuffer())
                st.session_state['active_dem'] = f.name
                st.rerun()

    # Tombol Reset / Ganti File
    if 'active_dem' in st.session_state:
        st.success("✅ File DEM Siap")
        if st.button("❌ Ganti File"):
            del st.session_state['active_dem']
            if 'engine' in st.session_state:
                del st.session_state['engine']
            st.rerun()
            
    st.markdown("---")
    btn_proc = st.button("🌊 PROSES FLOW DIRECTION")

with c2:
    # Proses Analisis
    if btn_proc and 'active_dem' in st.session_state:
        with st.spinner("Menganalisis Arah Aliran..."):
            try:
                eng = HydroEngine(st.session_state['active_dem'])
                eng.condition_dem()
                st.session_state['engine'] = eng
                st.success("✅ Selesai! Klik peta untuk delineasi.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Peta Interaktif
    if 'active_dem' in st.session_state:
        m = leafmap.Map()
        try:
            m.add_raster(st.session_state['active_dem'], layer_name="DEM", colormap="terrain")
        except: pass
        
        out = st_folium(m, height=500, width=None)
        
        # Delineasi saat klik
        if out and out['last_clicked'] and 'engine' in st.session_state:
            lat = out['last_clicked']['lat']
            lng = out['last_clicked']['lng']
            st.info(f"Titik Klik: {lat}, {lng}")
            
            with st.spinner("Delineasi DAS..."):
                poly = st.session_state['engine'].delineate(lng, lat)
                if poly:
                    area = poly.area * 12391 # Estimasi kasar konversi ke km2
                    st.success(f"✅ Luas DAS: {area:.2f} km² (Estimasi)")
                    
                    # Download
                    gdf_res = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
                    st.download_button("📥 Download GeoJSON", gdf_res.to_json(), "das.geojson")
                else:
                    st.warning("Gagal delineasi. Pastikan klik tepat di alur sungai.")
