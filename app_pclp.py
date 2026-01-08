# ==============================================================================
# SMART PCLP STUDIO PRO - INTEGRATED VERSION (FINAL FIX)
# Perbaikan: HydroEngine Direct-Array & Leafmap Localtileserver
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import math
import matplotlib.pyplot as plt

# --- 1. IMPORT LIBRARY ---
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
    st.error(f"⚠️ Error Import: {e}. Cek requirements.txt.")

try:
    import ezdxf
except ImportError:
    pass

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Smart PCLP Studio v7.1", layout="wide", page_icon="🚜")

# ==============================================================================
# BAGIAN A: MESIN HIDROLOGI (DIRECT ARRAY METHOD)
# ==============================================================================

class HydroEngine:
    """
    Mesin analisis hidrologi Pysheds.
    REVISI: Menggunakan passing objek Raster langsung (bukan via string registry)
    agar kompatibel dengan semua versi pysheds.
    """
    def __init__(self, dem_path):
        # 1. Inisialisasi Grid Geometry
        self.grid = Grid.from_raster(dem_path)
        # 2. Baca Data Elevasi sebagai Raster Object
        self.dem = self.grid.read_raster(dem_path)
        
        # Variabel untuk menyimpan hasil olahan
        self.fdir = None 
        self.acc = None

    def condition_dem(self):
        """Memproses Flow Direction & Accumulation"""
        # Fill Depressions (Langsung pada objek self.dem)
        # resolve_flats & fill_depressions mengembalikan array baru
        flooded_dem = self.grid.fill_depressions(self.dem)
        inflated_dem = self.grid.resolve_flats(flooded_dem)
        
        # Flow Direction (D8)
        # Mapping arah: N, NE, E, SE, S, SW, W, NW
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        
        # Simpan fdir ke self.fdir agar bisa dipakai delineasi nanti
        self.fdir = self.grid.flowdir(inflated_dem, dirmap=dirmap)
        
        # Flow Accumulation
        self.acc = self.grid.accumulation(self.fdir, dirmap=dirmap)
        
        return self.acc

    def delineate_catchment(self, x, y):
        """Mendelineasi DAS"""
        if self.fdir is None or self.acc is None:
            return None
            
        xy = (x, y)
        try:
            # Snap titik klik (Menggunakan array self.acc langsung)
            snapped_xy = self.grid.snap_to_mask(self.acc > 100, xy)
            
            # Catchment (Menggunakan array self.fdir langsung)
            # Perhatikan argumen 'fdir=' digunakan eksplisit
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
                if value == 1: # Nilai 1 = Catchment area
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
    """Download DEM Copernicus"""
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
        st.error(f"Gagal Download DEM: {e}")
        return None

# ==============================================================================
# BAGIAN B: MESIN SIPIL (PCLP)
# ==============================================================================
def parse_pclp_block(df):
    parsed_data = []
    i = 0
    df = df.astype(str)
    while i < len(df):
        row = df.iloc[i].values
        x_indices = [idx for idx, val in enumerate(row) if str(val).strip().upper() == 'X']
        if x_indices and (i + 1 < len(df)):
            x_idx = x_indices[0]
            val_y = str(df.iloc[i+1, x_idx]).strip().upper()
            if val_y == 'Y':
                sta_name = f"STA_{len(parsed_data)}"
                candidate = str(df.iloc[i+1, 1]).strip()
                if candidate.lower() not in ['nan', 'none', '']:
                    sta_name = candidate.replace('.0', '')
                
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
# BAGIAN C: UI UTAMA
# ==============================================================================
st.sidebar.title("🎛️ Menu Utama")
pilihan_menu = st.sidebar.radio("Pilih Modul:", ["1. Analisis Hidrologi (DAS)", "2. Desain Sipil (Cross/Long)"])

if pilihan_menu == "1. Analisis Hidrologi (DAS)":
    st.title("💧 Analisis Hidrologi & Delineasi DAS")
    st.info("Pastikan upload KML/GeoJSON dan tunggu proses download DEM.")

    # 1. Upload Batas Wilayah
    up_file = st.file_uploader("Upload Batas Wilayah (GeoJSON/KML)", type=["geojson", "kml", "kmz"])
    
    if up_file:
        try:
            fiona.drvsupport.supported_drivers['KML'] = 'rw'
            fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
        except: pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_file.name.split('.')[-1]}") as tmp:
            tmp.write(up_file.getbuffer())
            tmp_path = tmp.name

        try:
            gdf = gpd.read_file(tmp_path)
            if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            st.session_state['bbox'] = gdf.total_bounds
            st.success(f"Wilayah dimuat! Bound: {st.session_state['bbox']}")
        except Exception as e: st.error(f"Gagal baca file: {e}")

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
        st.subheader("Peta & Delineasi")
        st.caption("Klik area lembah/sungai untuk membuat DAS.")
        
        m = leafmap.Map()
        # Menggunakan add_raster dengan localtileserver (perlu diinstall)
        try:
            m.add_raster(st.session_state['dem_path'], layer_name="DEM", colormap="terrain")
        except Exception as e:
            st.warning(f"Gagal memuat raster overlay: {e}. Pastikan 'localtileserver' ada di requirements.txt")
        
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
            ax.legend(); ax.grid(True)
            st.pyplot(fig)
            dxf_btn = generate_dxf(res, "cross")
            st.download_button("📥 Download DXF", dxf_btn, "cross_section.dxf")
            
    with tab2:
        st.subheader("Long Section")
        f_long = st.file_uploader("Upload Long Section (.csv)", type=['csv'])
        if f_long:
            df = pd.read_csv(f_long)
            pts = df.iloc[:, :2].dropna().values.tolist()
            st.session_state['long_pts'] = pts
            
        if 'long_pts' in st.session_state:
            pts = st.session_state['long_pts']
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(*zip(*pts), 'b-')
            st.pyplot(fig)
            dxf_long = generate_dxf((pts, []), "long")
            st.download_button("📥 Download DXF Long", dxf_long, "long_section.dxf")
