import streamlit as st
import tempfile
import os

# --- IMPORT LIBRARY GEOSPASIAL ---
try:
    import geopandas as gpd
    import rasterio
    from pysheds.grid import Grid
    import leafmap.foliumap as leafmap
    from streamlit_folium import st_folium
    from shapely.geometry import Polygon
    import fiona
except ImportError as e:
    st.error(f"⚠️ Modul Hidrologi kurang lengkap: {e}")

# ==========================================
# 1. HYDRO ENGINE
# ==========================================
class HydroEngine:
    def __init__(self, dem_path):
        self.grid = Grid.from_raster(dem_path)
        self.dem = self.grid.read_raster(dem_path)
        self.grid.add_gridded_data(self.dem, data_name='dem')
        self.dem_view = self.grid.view('dem')
        self.fdir = None 
        self.acc = None

    def condition_dem(self):
        flooded_dem = self.grid.fill_depressions(self.dem)
        inflated_dem = self.grid.resolve_flats(flooded_dem)
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        self.fdir = self.grid.flowdir(inflated_dem, dirmap=dirmap)
        self.acc = self.grid.accumulation(self.fdir, dirmap=dirmap)
        return self.acc

    def delineate_catchment(self, x, y):
        if self.fdir is None: return None
        xy = (x, y)
        try:
            snapped_xy = self.grid.snap_to_mask(self.acc > 100, xy)
            catch = self.grid.catchment(x=snapped_xy[0], y=snapped_xy[1], 
                                       fdir=self.fdir, 
                                       dirmap=(64, 128, 1, 2, 4, 8, 16, 32), 
                                       xytype='coordinate')
            self.grid.clip_to(catch)
            shapes = self.grid.polygonize()
            catchment_poly = None
            max_area = 0
            for shape, value in shapes:
                if value == 1:
                    poly = Polygon(shape['coordinates'][0])
                    if poly.area > max_area:
                        max_area = poly.area
                        catchment_poly = poly
            return catchment_poly
        except: return None

# ==========================================
# 2. UI HALAMAN HIDROLOGI
# ==========================================
st.title("💧 Analisis Hidrologi & DAS")
st.caption("Modul Delineasi Daerah Aliran Sungai (DAS).")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Sumber Data DEM")
    
    # --- LOGIKA "AMBIL / KIRIM" DATA (MANUAL TRIGGER) ---
    dem_source = None
    using_pclp = False
    
    # 1. Cek apakah ada data di "Keranjang" Session State (dari App PCLP)
    has_pclp_data = 'gis_files' in st.session_state and st.session_state['gis_files'][0] is not None
    
    # 2. Status Penggunaan Data PCLP (Disimpan biar gak ilang saat klik lain)
    if 'use_pclp_data' not in st.session_state:
        st.session_state['use_pclp_data'] = False

    # 3. Tampilan Kontrol
    if has_pclp_data:
        if not st.session_state['use_pclp_data']:
            st.info("📦 Terdeteksi DEM dari Modul Sipil.")
            if st.button("🔄 Ambil Data dari Modul Sipil"):
                st.session_state['use_pclp_data'] = True
                st.rerun() # Refresh agar UI update
        else:
            st.success("✅ Menggunakan DEM dari Modul Sipil")
            dem_source = st.session_state['gis_files'][0]
            using_pclp = True
            
            if st.button("❌ Batal / Upload Manual"):
                st.session_state['use_pclp_data'] = False
                st.rerun()
    
    # 4. Jika Tidak Pakai Data PCLP, Tampilkan Upload Manual
    if not using_pclp:
        dem_source = st.file_uploader("Upload File DEM (.tif)", type=['tif', 'tiff'])

    st.markdown("---")
    # A. INPUT BATAS WILAYAH (Opsional)
    up_boundary = st.file_uploader("Batas Wilayah (Opsional - KML/GeoJSON)", type=["geojson", "kml", "kmz"])

    # TOMBOL PROSES
    btn_process = st.button("🌊 MULAI ANALISIS HIDROLOGI")

with col2:
    # LOGIKA PEMROSESAN
    if btn_process and dem_source:
        with st.spinner("Sedang memproses DEM (Flow Direction)..."):
            try:
                # Reset pointer file
                dem_source.seek(0)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f_dem:
                    f_dem.write(dem_source.read())
                    dem_path = f_dem.name
                
                # Simpan path
                st.session_state['hydro_dem_path'] = dem_path
                
                # Jalankan Engine
                eng = HydroEngine(dem_path)
                eng.condition_dem()
                st.session_state['hydro_engine'] = eng
                st.success("✅ Analisis Selesai! Silakan klik peta.")
                
            except Exception as e:
                st.error(f"Gagal memproses: {e}")

    # TAMPILKAN PETA
    if 'hydro_dem_path' in st.session_state:
        st.subheader("2. Peta Interaktif")
        
        m = leafmap.Map()
        
        try:
            m.add_raster(st.session_state['hydro_dem_path'], layer_name="Topografi", colormap="terrain")
        except:
            st.warning("Gagal visualisasi raster, tapi data siap didelineasi.")

        if up_boundary:
            try:
                fiona.drvsupport.supported_drivers['KML'] = 'rw'
                fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_boundary.name.split('.')[-1]}") as tmp:
                    tmp.write(up_boundary.getbuffer())
                    tmp_path = tmp.name
                gdf = gpd.read_file(tmp_path)
                bounds = gdf.total_bounds
                m.zoom_to_bounds((bounds[0], bounds[1], bounds[2], bounds[3]))
                style = {'fillColor': '#00000000', 'color': 'red', 'weight': 2}
                m.add_gdf(gdf, layer_name="Batas", style=style)
            except: pass
            
        map_out = st_folium(m, height=500, width=None)
        
        if map_out and map_out['last_clicked']:
            lat = map_out['last_clicked']['lat']
            lng = map_out['last_clicked']['lng']
            st.info(f"📍 Klik: {lat:.5f}, {lng:.5f}")
            
            if 'hydro_engine' in st.session_state:
                with st.spinner("Menghitung DAS..."):
                    eng = st.session_state['hydro_engine']
                    poly_das = eng.delineate_catchment(lng, lat)
                    
                    if poly_das:
                        area_km2 = (poly_das.area * 111.32 * 111.32)
                        st.success(f"✅ DAS Terbentuk! Luas: {area_km2:.2f} km²")
                        
                        gdf_res = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                        st.download_button("📥 Download GeoJSON", gdf_res.to_json(), "das_result.geojson")
                    else:
                        st.warning("⚠️ Gagal. Klik tepat di alur sungai.")
    else:
        st.info("👈 Upload/Ambil DEM dan Klik Tombol Proses.")
