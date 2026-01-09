import streamlit as st
import tempfile
import os

# --- IMPORT LIBRARY GEOSPASIAL ---
# Menggunakan try-except agar tidak error fatal
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
    from shapely.geometry import Polygon
    import fiona
except ImportError as e:
    st.error(f"⚠️ Modul Hidrologi kurang lengkap: {e}")

# CATATAN: st.set_page_config SUDAH DIHAPUS (Diatur oleh main.py)

# ==========================================
# 1. HYDRO ENGINE (MESIN HIDROLOGI)
# ==========================================
class HydroEngine:
    """Mesin analisis hidrologi Pysheds (Versi Stabil)."""
    def __init__(self, dem_path):
        # 1. Inisialisasi Grid dari file
        self.grid = Grid.from_raster(dem_path)
        # 2. Baca Data Elevasi
        self.dem = self.grid.read_raster(dem_path)
        # 3. Masukkan data ke Grid (Tanpa parameter affine/crs manual)
        self.grid.add_gridded_data(self.dem, data_name='dem')
        # 4. Set view
        self.dem_view = self.grid.view('dem')
        self.fdir = None 
        self.acc = None

    def condition_dem(self):
        """Memproses Flow Direction & Accumulation"""
        # Fill Depressions
        flooded_dem = self.grid.fill_depressions(self.dem)
        inflated_dem = self.grid.resolve_flats(flooded_dem)
        
        # Flow Direction (D8)
        # Mapping: N, NE, E, SE, S, SW, W, NW
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        self.fdir = self.grid.flowdir(inflated_dem, dirmap=dirmap)
        
        # Flow Accumulation
        self.acc = self.grid.accumulation(self.fdir, dirmap=dirmap)
        return self.acc

    def delineate_catchment(self, x, y):
        """Mendelineasi DAS"""
        if self.fdir is None: return None
        xy = (x, y)
        try:
            # Snap titik klik ke sungai terdekat (Acc > 100)
            snapped_xy = self.grid.snap_to_mask(self.acc > 100, xy)
            
            # Catchment
            catch = self.grid.catchment(x=snapped_xy[0], y=snapped_xy[1], 
                                       fdir=self.fdir, 
                                       dirmap=(64, 128, 1, 2, 4, 8, 16, 32), 
                                       xytype='coordinate')
            
            # Konversi ke Polygon
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
# 2. FUNGSI DOWNLOAD DEM
# ==========================================
@st.cache_data
def fetch_dem_copernicus(bbox):
    """Download DEM Copernicus dari Microsoft Planetary Computer"""
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

        dem_arrays = []
        for item in items:
            signed_href = item.assets["data"].href
            da = rioxarray.open_rasterio(signed_href).rio.clip_box(*bbox)
            dem_arrays.append(da)
            
        if len(dem_arrays) > 1:
            return merge_arrays(dem_arrays)
        return dem_arrays[0]
    except Exception as e:
        return None

# ==========================================
# 3. UI HALAMAN HIDROLOGI
# ==========================================
st.title("💧 Analisis Hidrologi & DAS")
st.info("Modul ini untuk mendelineasi Daerah Aliran Sungai (DAS) otomatis.")

# 1. Upload Batas Wilayah
up_file = st.file_uploader("1. Upload Batas Wilayah (GeoJSON/KML)", type=["geojson", "kml", "kmz"])

if up_file:
    # Handling Driver KML
    try:
        fiona.drvsupport.supported_drivers['KML'] = 'rw'
        fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
    except: pass

    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_file.name.split('.')[-1]}") as tmp:
        tmp.write(up_file.getbuffer())
        tmp_path = tmp.name

    try:
        gdf = gpd.read_file(tmp_path)
        if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
        st.session_state['bbox'] = gdf.total_bounds
        st.success(f"Wilayah dimuat! Bound: {st.session_state['bbox']}")
    except Exception as e:
        st.error(f"Gagal baca file: {e}")

# 2. Download DEM
if 'bbox' in st.session_state:
    if st.button("⬇️ Download & Proses DEM"):
        with st.spinner("Mengunduh DEM..."):
            dem_xr = fetch_dem_copernicus(st.session_state['bbox'])
            if dem_xr is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f_dem:
                    dem_xr.rio.to_raster(f_dem.name)
                    st.session_state['dem_path'] = f_dem.name
                
                st.success("DEM Berhasil Diunduh!")
                try:
                    eng = HydroEngine(st.session_state['dem_path'])
                    eng.condition_dem()
                    st.session_state['hydro_engine'] = eng
                    st.success("Analisis Arah Aliran Selesai!")
                except Exception as e: st.error(f"Error Hydro Engine: {e}")

# 3. Peta Interaktif
if 'dem_path' in st.session_state:
    st.subheader("3. Peta & Delineasi")
    st.caption("Klik area lembah/sungai untuk membuat DAS.")
    
    m = leafmap.Map()
    try:
        # Gunakan layer standar agar ringan (hindari localtileserver jika bermasalah)
        m.add_raster(st.session_state['dem_path'], layer_name="DEM", colormap="terrain")
    except:
        st.warning("Gagal memuat visualisasi raster, tapi delineasi tetap bisa jalan.")
    
    map_out = st_folium(m, height=500, width=None)
    
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
                    gdf_res = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                    st.download_button("📥 Download GeoJSON DAS", gdf_res.to_json(), "das_result.geojson")
                else:
                    st.warning("Gagal delineasi. Coba klik lebih pas di alur sungai.")
