import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import matplotlib.pyplot as plt
import tempfile
from shapely.geometry import Polygon, LineString

# --- IMPORT LIBRARY TAMBAHAN ---
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
except ImportError:
    st.warning("⚠️ Library 'ezdxf' belum terinstall.")

HAS_GEO_LIBS = False
try:
    import geopandas as gpd
    import rasterio
    HAS_GEO_LIBS = True
except ImportError:
    pass

# ==========================================
# 1. PARSER & LOGIKA SIPIL (CROSS SECTION)
# ==========================================
def parse_pclp_block(df):
    """Parser data Excel Cross Section"""
    parsed_data = []
    i = 0
    df = df.astype(str)
    
    while i < len(df):
        row = df.iloc[i].values
        x_indices = [idx for idx, val in enumerate(row) if val.strip().upper() == 'X']
        
        if x_indices and (i + 1 < len(df)):
            x_idx = x_indices[0]
            val_y = str(df.iloc[i+1, x_idx]).strip().upper()
            
            if val_y == 'Y':
                sta_name = f"STA_{len(parsed_data)}"
                candidate_sta = str(df.iloc[i+1, 1]).strip() 
                if candidate_sta.lower() not in ['nan', 'none', '']:
                    sta_name = candidate_sta
                if sta_name.endswith('.0'): sta_name = sta_name[:-2]

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

def hitung_cut_fill(tanah_pts, desain_pts):
    if not tanah_pts or not desain_pts: return 0.0, 0.0
    min_y = min([p[1] for p in tanah_pts] + [p[1] for p in desain_pts])
    datum = min_y - 5.0
    p_tanah = tanah_pts + [(tanah_pts[-1][0], datum), (tanah_pts[0][0], datum)]
    p_desain = desain_pts + [(desain_pts[-1][0], datum), (desain_pts[0][0], datum)]
    
    try:
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
            col = i % 2; row = i // 2
            ox, oy = col * 60, row * -40
            t_pts = [(p[0]+ox, p[1]+oy) for p in item.get('points_tanah', [])]
            d_pts = [(p[0]+ox, p[1]+oy) for p in item.get('points_desain', [])]
            
            if t_pts: msp.add_lwpolyline(t_pts, dxfattribs={'layer': 'TANAH'})
            if d_pts: msp.add_lwpolyline(d_pts, dxfattribs={'layer': 'DESAIN'})
            
            txt = f"{item['STA']}"
            if d_pts: txt += f" | C:{item['cut']:.2f} | F:{item['fill']:.2f}"
            msp.add_text(txt, dxfattribs={'height':0.5, 'layer':'TEXT'}).set_placement((ox, oy))
            
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

# ==========================================
# 2. FUNGSI INTEGRASI GIS
# ==========================================
def extract_long_section(dem_path, shp_path, interval):
    if not HAS_GEO_LIBS: return None
    try:
        with rasterio.open(dem_path) as src:
            gdf = gpd.read_file(shp_path)
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            line = gdf.geometry.iloc[0]
            if line.geom_type == 'MultiLineString': line = line.geoms[0]
            
            data = []
            for dist in np.arange(0, line.length, interval):
                pt = line.interpolate(dist)
                try:
                    for val in src.sample([(pt.x, pt.y)]):
                        if val[0] != src.nodata:
                            data.append([dist, val[0]])
                except: pass
            return data
    except: return None

def render_preview(dem_path, shp_path):
    try:
        with rasterio.open(dem_path) as src:
            gdf = gpd.read_file(shp_path)
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            # Downsample for preview speed
            data = src.read(1, out_shape=(src.height//10, src.width//10))
            data = np.ma.masked_where(data == src.nodata, data)
            
            ax.imshow(data, cmap='terrain', extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))
            gdf.plot(ax=ax, color='red', linewidth=2)
            ax.set_title("Preview Peta Situasi")
            ax.axis('off')
            return fig
    except: return None

# ==========================================
# 3. USER INTERFACE (MAIN)
# ==========================================
st.title("🚜 PCLP Studio Pro v6.1 (Stable)")

tabs = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🗺️ PETA SITUASI (GIS)"])

# --- TAB 1: CROSS SECTION ---
with tabs[0]:
    col_in, col_view = st.columns([1, 2])
    with col_in:
        st.subheader("Input Data")
        f_upload = st.file_uploader("Upload Excel (.xls/.xlsx)", type=['xls', 'xlsx'], key='cross_up')
        
        if f_upload:
            try:
                xls = pd.ExcelFile(f_upload)
                s_ogl = st.selectbox("Sheet Tanah", ["[Pilih]"] + xls.sheet_names)
                s_dsn = st.selectbox("Sheet Desain", ["[Pilih]"] + xls.sheet_names)
                
                if st.button("PROSES DATA CROSS"):
                    d_ogl = parse_pclp_block(pd.read_excel(f_upload, sheet_name=s_ogl, header=None)) if s_ogl != "[Pilih]" else []
                    d_dsn = parse_pclp_block(pd.read_excel(f_upload, sheet_name=s_dsn, header=None)) if s_dsn != "[Pilih]" else []
                    
                    final = []
                    for i in range(max(len(d_ogl), len(d_dsn))):
                        t = d_ogl[i] if i < len(d_ogl) else None
                        d = d_dsn[i] if i < len(d_dsn) else None
                        sta = t['STA'] if t else (d['STA'] if d else f"STA_{i}")
                        tp = t['points'] if t else []
                        dp = d['points'] if d else []
                        c, f = hitung_cut_fill(tp, dp)
                        final.append({'STA': sta, 'points_tanah': tp, 'points_desain': dp, 'cut': c, 'fill': f})
                    
                    st.session_state['data_cross'] = final
                    st.success("Selesai!")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_view:
        if 'data_cross' in st.session_state:
            data = st.session_state['data_cross']
            if data:
                idx = st.slider("Pilih STA", 0, len(data)-1, 0)
                item = data[idx]
                
                fig, ax = plt.subplots(figsize=(10, 3))
                if item['points_tanah']: ax.plot(*zip(*item['points_tanah']), 'k-o', label='Tanah')
                if item['points_desain']: ax.plot(*zip(*item['points_desain']), 'r-', label='Desain')
                ax.set_title(f"{item['STA']} | Cut:{item['cut']:.2f} | Fill:{item['fill']:.2f}")
                ax.legend(); ax.grid(True)
                st.pyplot(fig)
                
                dxf = generate_dxf(data, "cross")
                st.download_button("📥 Download DXF", dxf, "Cross.dxf")

# --- TAB 2: LONG SECTION ---
with tabs[1]:
    st.subheader("Long Section Profile")
    
    # Cek Data Kiriman dari GIS
    if 'long_res' in st.session_state:
        st.info("✅ Menampilkan Data dari Tab Peta Situasi")
        ogl, dsn = st.session_state['long_res']
        
        fig, ax = plt.subplots(figsize=(12, 4))
        if ogl: ax.plot(*zip(*ogl), 'k--', label='Tanah Asli')
        if dsn: ax.plot(*zip(*dsn), 'r-', label='Desain')
        ax.grid(True); ax.legend()
        st.pyplot(fig)
        
        dxf_long = generate_dxf((ogl, dsn), "long")
        st.download_button("📥 Download DXF Long", dxf_long, "Long.dxf")
        
    else:
        st.caption("Belum ada data. Upload manual atau gunakan Tab Peta Situasi.")
        f_long = st.file_uploader("Upload CSV/Excel Manual", type=['csv', 'xlsx'])

# --- TAB 3: PETA SITUASI (INTEGRASI PENUH) ---
with tabs[2]:
    st.header("🗺️ Peta Situasi & Integrasi Data")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info("Upload Data untuk Integrasi")
        up_dem = st.file_uploader("Upload DEM (.tif)", type=['tif', 'tiff'])
        up_shp = st.file_uploader("Upload Trase (.geojson/.shp)", type=['geojson', 'shp'], accept_multiple_files=True)
        
        # --- LOGIKA AUTO-SAVE (PENTING) ---
        if up_dem:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f:
                f.write(up_dem.getbuffer())
                st.session_state['shared_dem_path'] = f.name
                st.toast("✅ DEM Tersimpan untuk Hidrologi")
        
        shp_path = None
        if up_shp:
            for f in up_shp:
                if f.name.endswith(('.geojson', '.shp')):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}") as tmp:
                        tmp.write(f.getbuffer())
                        shp_path = tmp.name
                        st.toast("✅ Trase Tersimpan")
                    break
        
        btn_render = st.button("1. Tampilkan Peta")
        btn_extract = st.button("2. Ekstrak ke Long Section")

    with c2:
        # Render Peta
        if btn_render and 'shared_dem_path' in st.session_state and shp_path:
            with st.spinner("Merender..."):
                fig = render_preview(st.session_state['shared_dem_path'], shp_path)
                if fig: st.pyplot(fig)
        
        # Ekstrak Data
        if btn_extract and 'shared_dem_path' in st.session_state and shp_path:
            with st.spinner("Mengekstrak elevasi..."):
                data_long = extract_long_section(st.session_state['shared_dem_path'], shp_path, 25)
                if data_long:
                    st.session_state['long_res'] = (data_long, [])
                    st.success(f"Berhasil ekstrak {len(data_long)} titik! Cek Tab Long Section.")
                else:
                    st.error("Gagal ekstrak data.")
