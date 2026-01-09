import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from shapely.geometry import Polygon, LineString

# --- HANDLING IMPORT LIBRARY ---
try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
except ImportError:
    st.warning("⚠️ Library 'ezdxf' belum terinstall. Fitur DXF tidak akan jalan.")

HAS_GEO_LIBS = False
try:
    import geopandas as gpd
    import rasterio
    from rasterio.plot import show
    HAS_GEO_LIBS = True
except ImportError:
    pass

# ==========================================
# 1. PARSER ENGINE (ROBUST)
# ==========================================
def parse_pclp_block(df):
    """Parser untuk format Excel Blok PCLP (Cross Section)."""
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
    poly_tanah = Polygon(p_tanah).buffer(0)
    poly_desain = Polygon(p_desain).buffer(0)
    try:
        return poly_desain.intersection(poly_tanah).area, poly_desain.difference(poly_tanah).area
    except: return 0.0, 0.0

# ==========================================
# 2. GENERATOR OUTPUT
# ==========================================
def generate_dxf(results, mode="cross"):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add(name='TANAH', color=8)
    doc.layers.add(name='DESAIN', color=1)
    doc.layers.add(name='TEXT', color=7)
    doc.layers.add(name='GRID', color=9)

    if mode == "long":
        tanah, desain = results
        if tanah: msp.add_lwpolyline(tanah, dxfattribs={'layer': 'TANAH'})
        if desain: msp.add_lwpolyline(desain, dxfattribs={'layer': 'DESAIN'})
        if tanah:
            min_x, max_x = min(p[0] for p in tanah), max(p[0] for p in tanah)
            min_y, max_y = min(p[1] for p in tanah), max(p[1] for p in tanah)
            msp.add_line((min_x, min_y), (max_x, min_y), dxfattribs={'layer': 'GRID'})
            msp.add_text("LONG SECTION PROFILE", dxfattribs={'height': 2.0, 'layer': 'TEXT'}).set_placement((min_x, max_y + 5))
    else:
        for i, item in enumerate(results):
            col = i % 2; row = i // 2
            offset_x = col * 60; offset_y = row * -40
            t_pts = [(p[0]+ox, p[1]+oy) for p, ox, oy in [(pt, offset_x, offset_y) for pt in item.get('points_tanah', [])]]
            d_pts = [(p[0]+ox, p[1]+oy) for p, ox, oy in [(pt, offset_x, offset_y) for pt in item.get('points_desain', [])]]
            
            if t_pts: msp.add_lwpolyline(t_pts, dxfattribs={'layer': 'TANAH'})
            if d_pts: msp.add_lwpolyline(d_pts, dxfattribs={'layer': 'DESAIN'})
            
            info_txt = f"{item['STA']} | C:{item['cut']:.2f} | F:{item['fill']:.2f}"
            msp.add_text(info_txt, dxfattribs={'height': 0.5, 'layer': 'TEXT'}).set_placement((offset_x, offset_y))

    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

def generate_excel_report(data):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        rekap = []
        for item in data:
            rekap.append({'STA': item['STA'], 'Cut Area (m2)': item['cut'], 'Fill Area (m2)': item['fill']})
        pd.DataFrame(rekap).to_excel(writer, sheet_name='Volume Report', index=False)
    return output.getvalue()

# ==========================================
# 3. GEOSPATIAL ENGINE (EXTRACTION)
# ==========================================
def extract_long_section_from_dem(dem_file, shp_file, interval=25): # UPDATE: Default jadi 25
    """
    Mengambil data elevasi dari DEM berdasarkan garis shapefile.
    Output: DataFrame (Jarak, Z, X, Y)
    """
    if not HAS_GEO_LIBS: return None, "Library GIS Missing"
    
    try:
        with rasterio.open(dem_file) as src:
            gdf = gpd.read_file(shp_file)
            
            # Reproyeksi Shapefile agar sama dengan DEM
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            
            # Ambil geometri garis pertama
            line = gdf.geometry.iloc[0]
            if line.geom_type == 'MultiLineString':
                line = line.geoms[0] # Ambil bagian pertama saja jika multiline
                
            length = line.length
            points_data = []
            
            # Loop sepanjang garis (Stationing)
            for dist in np.arange(0, length, interval):
                pt = line.interpolate(dist)
                
                # Sampling Elevasi dari DEM
                # src.sample butuh list of (x, y)
                try:
                    for val in src.sample([(pt.x, pt.y)]):
                        elev = val[0]
                        if elev == src.nodata: elev = np.nan
                        points_data.append({
                            'Station (m)': dist,
                            'Elevation (m)': elev,
                            'X (Easting)': pt.x,
                            'Y (Northing)': pt.y
                        })
                except:
                    points_data.append({'Station (m)': dist, 'Elevation (m)': np.nan, 'X': pt.x, 'Y': pt.y})

            return pd.DataFrame(points_data), None
    except Exception as e:
        return None, str(e)

def render_peta_situasi(dem_file, shp_file):
    # (Fungsi rendering visual saja, sama seperti sebelumnya)
    if not HAS_GEO_LIBS: return None, "No GIS Libs"
    try:
        with rasterio.open(dem_file) as src:
            gdf = gpd.read_file(shp_file)
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            data = src.read(1, out_shape=(src.height//5, src.width//5))
            data_masked = np.ma.masked_where(data == src.nodata, data)
            x = np.linspace(src.bounds.left, src.bounds.right, data.shape[1])
            y = np.linspace(src.bounds.top, src.bounds.bottom, data.shape[0])
            X, Y = np.meshgrid(x, y)
            
            contours = ax.contour(X, Y, data_masked, levels=20, cmap='terrain', linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=6, fmt='%1.0f')
            gdf.plot(ax=ax, color='red', linewidth=2, label='Trase', zorder=5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title("Peta Situasi: Kontur & Trase")
            return fig, None
    except Exception as e: return None, str(e)


# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🚜 PCLP Studio Pro v6.1 (Stable)")
st.caption("Aplikasi Desain Irigasi & Jalan: Cross Section, Long Section & GIS Situasi")

if not HAS_GEO_LIBS:
    st.warning("⚠️ Modul Geospasial tidak aktif.")

tabs = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🗺️ PETA SITUASI (GIS)"])

# --- TAB 1: CROSS SECTION ---
with tabs[0]:
    col_in, col_view = st.columns([1, 2])
    with col_in:
        st.subheader("Input Data PCLP")
        f_upload = st.file_uploader("Upload Excel", type=['xls', 'xlsx'], key='cross_up')
        if f_upload:
            try:
                xls = pd.ExcelFile(f_upload)
                s_ogl = st.selectbox("Sheet Tanah", ["[Pilih]"]+xls.sheet_names)
                s_dsn = st.selectbox("Sheet Desain", ["[Pilih]"]+xls.sheet_names)
                if st.button("PROSES DATA"):
                    d_ogl = parse_pclp_block(pd.read_excel(f_upload, sheet_name=s_ogl, header=None)) if s_ogl != "[Pilih]" else []
                    d_dsn = parse_pclp_block(pd.read_excel(f_upload, sheet_name=s_dsn, header=None)) if s_dsn != "[Pilih]" else []
                    final = []
                    for i in range(max(len(d_ogl), len(d_dsn))):
                        t = d_ogl[i] if i < len(d_ogl) else None
                        d = d_dsn[i] if i < len(d_dsn) else None
                        sta = t['STA'] if t else (d['STA'] if d else f"STA_{i}")
                        tp, dp = (t['points'] if t else []), (d['points'] if d else [])
                        c, f = hitung_cut_fill(tp, dp)
                        final.append({'STA': sta, 'points_tanah': tp, 'points_desain': dp, 'cut': c, 'fill': f})
                    st.session_state['data_cross'] = final
                    st.success("Selesai!")
            except: st.error("Gagal baca file.")

    with col_view:
        if 'data_cross' in st.session_state:
            data = st.session_state['data_cross']
            idx = st.slider("Pilih STA", 0, len(data)-1, 0)
            item = data[idx]
            fig, ax = plt.subplots(figsize=(10, 4))
            if item['points_tanah']: ax.plot(*zip(*item['points_tanah']), 'k-o', label='Tanah')
            if item['points_desain']: ax.plot(*zip(*item['points_desain']), 'r-', label='Desain')
            ax.set_title(f"{item['STA']} | C:{item['cut']:.2f} | F:{item['fill']:.2f}")
            ax.legend(); ax.grid(True); st.pyplot(fig)
            
            c1, c2 = st.columns(2)
            c1.download_button("📥 DXF Cross", generate_dxf(data, "cross"), "Cross.dxf")
            c2.download_button("📥 Excel Report", generate_excel_report(data), "Vol_Report.xlsx")

# --- TAB 2: LONG SECTION ---
with tabs[1]:
    st.subheader("Long Section")
    f_long = st.file_uploader("Upload Long", type=['xls', 'xlsx', 'csv'], key='long_up')
    if f_long:
        try:
            df = pd.read_csv(f_long) if f_long.name.endswith('.csv') else pd.read_excel(f_long)
            st.session_state['long_res'] = (df.iloc[:, :2].dropna().values.tolist(), [])
            st.success("Data masuk!")
        except: st.error("Error file.")

    if 'long_res' in st.session_state:
        ogl, _ = st.session_state['long_res']
        # Tampilkan Tabel Data untuk dicek user
        with st.expander("Lihat Data Tabel"):
            st.dataframe(pd.DataFrame(ogl, columns=["Jarak (m)", "Elevasi (m)"]))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(*zip(*ogl), 'k--', label='Tanah Asli')
        ax.grid(True); st.pyplot(fig)
        st.download_button("📥 DXF Long", generate_dxf((ogl, []), "long"), "Long.dxf")

# --- TAB 3: PETA SITUASI (UPDATE FITUR EXPORT) ---
with tabs[2]:
    st.header("🗺️ Peta Situasi & Ekstraksi Data")
    st.caption("Upload DEM & Trase, lalu ekstrak data elevasi untuk Long Section/Civil 3D.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        up_dem = st.file_uploader("Upload DEM (.tif)", type=['tif', 'tiff'])
        up_shp = st.file_uploader("Upload Trase (.geojson/.shp)", type=['geojson', 'shp'], accept_multiple_files=True)
        
        # Opsi Interval Sampling (UPDATED: Default 25m, Step 5m)
        interval = st.number_input("Interval Sampling (m)", min_value=5, max_value=1000, value=25, step=5)
        
        shp_file = None
        if up_shp:
            for f in up_shp:
                if f.name.endswith('.geojson') or f.name.endswith('.shp'): shp_file = f; break
        
        # TOMBOL EKSEKUSI
        btn_render = st.button("1. Tampilkan Peta")
        btn_extract = st.button("2. Ekstrak Data Elevasi (Long Section)")

    with c2:
        # LOGIKA RENDER PETA
        if btn_render and up_dem and shp_file:
            st.session_state['gis_files'] = (up_dem, shp_file)
        
        if 'gis_files' in st.session_state:
            dem, shp = st.session_state['gis_files']
            dem.seek(0); shp.seek(0)
            with st.spinner("Merender Peta..."):
                fig, err = render_peta_situasi(dem, shp)
                if fig: st.pyplot(fig)
                else: st.error(err)

        # LOGIKA EKSTRAKSI DATA & EXPORT CIVIL 3D
        if btn_extract and up_dem and shp_file:
            up_dem.seek(0); shp_file.seek(0)
            with st.spinner(f"Mengambil sampel elevasi setiap {interval}m..."):
                df_long, err = extract_long_section_from_dem(up_dem, shp_file, interval)
                
                if df_long is not None:
                    st.success(f"Berhasil mengekstrak {len(df_long)} titik!")
                    
                    # Tampilkan Preview Data
                    st.dataframe(df_long.head())
                    
                    # 1. DOWNLOAD UNTUK CIVIL 3D / EXCEL
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_long.to_excel(writer, index=False, sheet_name='Long Section Data')
                    
                    st.download_button(
                        label="📥 Download Excel (Format Civil 3D/Land Desktop)",
                        data=output.getvalue(),
                        file_name="Data_Elevasi_Trase.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # 2. KIRIM KE TAB LONG SECTION
                    # Konversi DataFrame ke List of Lists [[Jarak, Elevasi], ...]
                    long_data = df_long[['Station (m)', 'Elevation (m)']].dropna().values.tolist()
                    st.session_state['long_res'] = (long_data, []) # Simpan ke Session State
                    st.info("✅ Data otomatis dikirim ke Tab 'LONG SECTION'. Silakan cek tab sebelah!")
                    
                else:
                    st.error(f"Gagal ekstrak: {err}")
