import streamlit as st
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, LineString, Point
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import io
import numpy as np
import math

# Cek Library Geospasial
try:
    import geopandas as gpd
    import rasterio
    from rasterio.plot import show
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False

# ==========================================
# 1. PARSER ENGINE (DUAL MODE: BLOK & TABEL)
# ==========================================
def parse_pclp_block(df):
    """Parser Mode 1: Format Blok Horizontal (Ada Huruf X dan Y)."""
    parsed_data = []
    i = 0
    while i < len(df):
        row = df.iloc[i]
        x_col_idx = -1
        max_col = min(20, len(row))
        for c in range(max_col):
            try:
                if str(row[c]).strip().upper() == 'X':
                    x_col_idx = c
                    break
            except: continue
        
        if x_col_idx != -1 and (i + 1 < len(df)):
            try:
                val_y = str(df.iloc[i+1][x_col_idx]).strip().upper()
                if val_y == 'Y':
                    sta_name = f"STA_{i}"
                    if x_col_idx >= 2:
                        val_sta = str(df.iloc[i+1][x_col_idx - 2]).strip()
                        if val_sta and val_sta.lower() != 'nan': sta_name = val_sta
                    if sta_name.endswith('.0'): sta_name = sta_name[:-2]

                    start = x_col_idx + 1
                    max_len = min(len(row), len(df.iloc[i+1]))
                    x_vals = row[start:max_len].values
                    y_vals = df.iloc[i+1][start:max_len].values
                    
                    points = []
                    for x, y in zip(x_vals, y_vals):
                        try:
                            xf, yf = float(x), float(y)
                            if not (pd.isna(xf) or pd.isna(yf)): points.append((xf, yf))
                        except: continue
                    
                    if points:
                        points.sort(key=lambda p: p[0])
                        parsed_data.append({'STA': sta_name, 'points': points})
                    i += 1 
            except: pass
        i += 1
    return parsed_data

def parse_long_tabular(df):
    """Parser Mode 2: Format Tabel Panjang (Untuk Long Section)."""
    header_idx = -1
    col_dist_idx = -1
    col_elev_idx = -1
    
    # Scan baris awal untuk cari header
    for r in range(min(20, len(df))):
        row_vals = [str(v).lower() for v in df.iloc[r].values]
        if any("dist" in x for x in row_vals) and (any("elev" in x for x in row_vals) or any("o.g.l" in x for x in row_vals)):
            header_idx = r
            for c, val in enumerate(row_vals):
                if "cum" in val or "dist" in val: 
                     if col_dist_idx == -1: col_dist_idx = c
                if "o.g.l" in val or "bl" in val or "elev" in val:
                     if col_elev_idx == -1: col_elev_idx = c
            break
            
    if header_idx != -1 and col_dist_idx != -1 and col_elev_idx != -1:
        points = []
        for i in range(header_idx+1, len(df)):
            try:
                dist = df.iloc[i, col_dist_idx]
                elev = df.iloc[i, col_elev_idx]
                fd, fe = float(dist), float(elev)
                if not (pd.isna(fd) or pd.isna(fe)):
                    points.append((fd, fe))
            except: continue
        points.sort(key=lambda x: x[0])
        return [{'STA': 'Long_Section', 'points': points}]
    return []

def combined_parser(df):
    """Mencoba baca sebagai Blok, jika kosong coba baca sebagai Tabel."""
    res = parse_pclp_block(df)
    if not res:
        res = parse_long_tabular(df)
    return res

def hitung_cut_fill(tanah_pts, desain_pts):
    if not tanah_pts or not desain_pts: return 0, 0
    all_y = [p[1] for p in tanah_pts] + [p[1] for p in desain_pts]
    datum = min(all_y) - 5.0
    poly_tanah = Polygon(tanah_pts + [(tanah_pts[-1][0], datum), (tanah_pts[0][0], datum)]).buffer(0)
    poly_desain = Polygon(desain_pts + [(desain_pts[-1][0], datum), (desain_pts[0][0], datum)]).buffer(0)
    try:
        area_cut = poly_desain.intersection(poly_tanah).area
        area_fill = poly_desain.difference(poly_tanah).area
    except: area_cut, area_fill = 0, 0
    return area_cut, area_fill

# ==========================================
# 2. GIS ENGINE
# ==========================================
def sample_dem_at_point(src, x, y):
    try:
        vals = list(src.sample([(x, y)]))
        val = vals[0][0]
        if val == src.nodata: return None
        return float(val)
    except: return None

def generate_cross_from_gis(dem_file, align_file, interval, left_w, right_w):
    if not HAS_GEO_LIBS: return None, "Library Geospasial Missing."
    try:
        gdf = gpd.read_file(align_file)
        # Force Projected CRS (Meter)
        if gdf.crs and gdf.crs.is_geographic: gdf = gdf.to_crs(epsg=3857)
        
        line = gdf.geometry.iloc[0]
        length = line.length
        stations = np.arange(0, length, interval)
        pclp_data = [] 
        
        with rasterio.open(dem_file) as src:
            for dist in stations:
                pt_center = line.interpolate(dist)
                pt_ahead = line.interpolate(min(dist + 0.1, length))
                
                dx = pt_ahead.x - pt_center.x
                dy = pt_ahead.y - pt_center.y
                
                vec_len = math.sqrt(dx*dx + dy*dy)
                if vec_len == 0: continue
                nx, ny = -dy/vec_len, dx/vec_len
                
                offsets = np.arange(-left_w, right_w + 1, 1.0)
                points = []
                
                for off in offsets:
                    wx = pt_center.x + (nx * off)
                    wy = pt_center.y + (ny * off)
                    elev = sample_dem_at_point(src, wx, wy)
                    if elev is not None:
                        points.append((off, elev))
                
                if points:
                    pclp_data.append({'STA': f"STA {int(dist)}", 'points': points})
                    
        return pclp_data, None
    except Exception as e: return None, str(e)

def pclp_data_to_csv(data_list):
    lines = ["Generated by PCLP Studio,,,,,,,,,"]
    for i, item in enumerate(data_list):
        sta = item['STA']
        pts = item['points']
        
        row_x = ["", "", "Elv. Min", "X"] + [f"{x:.2f}" for x in [p[0] for p in pts]]
        lines.append(",".join(row_x))
        
        ys = [p[1] for p in pts]
        row_y = [i+1, sta, min(ys) if ys else 0, "Y"] + [f"{y:.3f}" for y in ys]
        lines.append(",".join([str(x) for x in row_y]))
        lines.append(",,,,,,,,,")
    return "\n".join(lines)

# ==========================================
# 3. DXF GENERATORS
# ==========================================
def generate_dxf_smart(results, mode="cross"):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    if 'TANAH' not in doc.layers: doc.layers.add(name='TANAH', color=8)
    if 'DESAIN' not in doc.layers: doc.layers.add(name='DESAIN', color=1)
    if 'TEXT' not in doc.layers: doc.layers.add(name='TEXT', color=7)
    
    if mode == "long":
        tanah, desain = results
        if tanah: msp.add_lwpolyline(tanah, dxfattribs={'layer': 'TANAH'})
        if desain: msp.add_lwpolyline(desain, dxfattribs={'layer': 'DESAIN'})
        start_x = tanah[0][0] if tanah else 0
        msp.add_mtext("LONG SECTION", dxfattribs={'char_height': 2.0, 'layer': 'TEXT'}).set_location(insert=(start_x, 10), attachment_point=ezdxf.const.MTEXT_TOP_LEFT)
    else:
        x_gap = 60
        count = 0
        for item in results:
            offset = (count * x_gap, 0)
            t_pts = item.get('points_tanah', [])
            d_pts = item.get('points_desain', [])
            
            if t_pts:
                draw_t = [(p[0]+offset[0], p[1]+offset[1]) for p in t_pts]
                msp.add_lwpolyline(draw_t, dxfattribs={'layer': 'TANAH'})
            if d_pts:
                draw_d = [(p[0]+offset[0], p[1]+offset[1]) for p in d_pts]
                msp.add_lwpolyline(draw_d, dxfattribs={'layer': 'DESAIN'})
            
            info = f"{item['STA']}\\P"
            if t_pts and d_pts:
                info += f"Cut: {item['cut']:.2f} m2\\PFill: {item['fill']:.2f} m2"
            
            cy = max([p[1] for p in t_pts] if t_pts else [0])
            cx = (draw_t[0][0] if t_pts else 0)
            msp.add_mtext(info, dxfattribs={'char_height':0.4, 'layer':'TEXT'}).set_location(insert=(cx, cy+4), attachment_point=ezdxf.const.MTEXT_TOP_CENTER)
            count += 1
            
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

# ==========================================
# 4. UI STREAMLIT
# ==========================================
st.set_page_config(page_title="PCLP Studio v5.1", layout="wide")
st.title("🚜 PCLP Studio v5.1 (Fixed & Complete)")

if not HAS_GEO_LIBS:
    st.error("⚠️ Library Geospasial tidak terdeteksi! Tab Peta Situasi tidak akan muncul. Install `geopandas rasterio`.")

tab1, tab2, tab3 = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🗺️ PETA SITUASI & GIS"])

# --- TAB 1: CROSS SECTION ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. Input Data")
        f_cross = st.file_uploader("Upload Excel Cross", key="uc")
        
        if f_cross:
            try:
                xls = pd.ExcelFile(f_cross)
                s_ogl = st.selectbox("Sheet Tanah:", ["(Tidak Ada)"] + xls.sheet_names, index=1)
                s_des = st.selectbox("Sheet Desain:", ["(Tidak Ada)"] + xls.sheet_names, index=min(2, len(xls.sheet_names)))
                
                if st.button("PROSES DATA"):
                    d_ogl, d_des = [], []
                    if s_ogl != "(Tidak Ada)": d_ogl = combined_parser(pd.read_excel(f_cross, sheet_name=s_ogl, header=None))
                    if s_des != "(Tidak Ada)": d_des = combined_parser(pd.read_excel(f_cross, sheet_name=s_des, header=None))
                    
                    final_res = []
                    limit = max(len(d_ogl), len(d_des))
                    for i in range(limit):
                        t = d_ogl[i] if i < len(d_ogl) else {'STA': f'STA_{i}', 'points': []}
                        d = d_des[i] if i < len(d_des) else {'STA': f'STA_{i}', 'points': []}
                        c, f = hitung_cut_fill(t['points'], d['points'])
                        final_res.append({'STA': t['STA'], 'cut': c, 'fill': f, 'points_tanah': t['points'], 'points_desain': d['points']})
                    
                    st.session_state['res_cross'] = final_res
                    st.session_state['sta_index'] = 0
                    st.success("Data Terbaca!")
            except Exception as e: st.error(f"Error: {e}")

    with col2:
        if 'res_cross' in st.session_state:
            data = st.session_state['res_cross']
            
            # Navigasi
            st.subheader("2. Preview & Koreksi")
            c_nav1, c_nav2, c_nav3 = st.columns([1, 2, 1])
            if c_nav1.button("◀ Prev"): st.session_state['sta_index'] = max(0, st.session_state['sta_index'] - 1)
            if c_nav3.button("Next ▶"): st.session_state['sta_index'] = min(len(data) - 1, st.session_state['sta_index'] + 1)
            
            current_idx = st.session_state['sta_index']
            selected_sta = c_nav2.selectbox("STA:", [d['STA'] for d in data], index=current_idx)
            
            # Sync Selectbox
            if [d['STA'] for d in data].index(selected_sta) != current_idx:
                 st.session_state['sta_index'] = [d['STA'] for d in data].index(selected_sta)
            
            item = data[st.session_state['sta_index']]
            
            # Offset Adjuster
            c_adj1, c_adj2 = st.columns(2)
            dx = c_adj1.number_input("Geser X (m):", 0.0, step=0.5, key='dx')
            dy = c_adj2.number_input("Geser Y (m):", 0.0, step=0.5, key='dy')
            
            t_pts = [(p[0]+dx, p[1]+dy) for p in item['points_tanah']]
            d_pts = item['points_desain']
            cut, fill = hitung_cut_fill(t_pts, d_pts)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            if t_pts: ax.plot(*zip(*t_pts), 'k-o', markersize=4, label='Tanah')
            if d_pts: ax.plot(*zip(*d_pts), 'r-o', markersize=4, label='Desain')
            ax.set_title(f"{item['STA']} | Cut: {cut:.2f} | Fill: {fill:.2f}")
            ax.legend(); ax.grid(True); ax.set_aspect('equal')
            st.pyplot(fig)
            
            dxf_bytes = generate_dxf_smart(data, mode="cross")
            st.download_button("📥 DOWNLOAD DXF (ALL)", dxf_bytes, "Cross_Result.dxf", "application/dxf")

# --- TAB 2: LONG SECTION ---
with tab2:
    st.subheader("Long Section Profile")
    f_long = st.file_uploader("Upload Excel Long", key="ul")
    if f_long:
        try:
            xls_l = pd.ExcelFile(f_long)
            sl_ogl = st.selectbox("Sheet Tanah:", ["(Tidak Ada)"] + xls_l.sheet_names, index=1, key="slo")
            sl_des = st.selectbox("Sheet Desain:", ["(Tidak Ada)"] + xls_l.sheet_names, index=min(2, len(xls_l.sheet_names)), key="sld")
            
            if st.button("RUN LONG SECTION"):
                mt, md = [], []
                # PENTING: Menggunakan combined_parser agar bisa baca tabel
                if sl_ogl != "(Tidak Ada)":
                    for p in combined_parser(pd.read_excel(f_long, sheet_name=sl_ogl, header=None)): mt.extend(p['points'])
                    mt.sort(key=lambda x: x[0])
                if sl_des != "(Tidak Ada)":
                    for p in combined_parser(pd.read_excel(f_long, sheet_name=sl_des, header=None)): md.extend(p['points'])
                    md.sort(key=lambda x: x[0])
                st.session_state['res_long'] = (mt, md)
        except Exception as e: st.error(f"Error: {e}")
        
    if 'res_long' in st.session_state:
        mt, md = st.session_state['res_long']
        fig, ax = plt.subplots(figsize=(10, 4))
        if mt: ax.plot(*zip(*mt), 'k-', label='Tanah')
        if md: ax.plot(*zip(*md), 'r-', label='Desain')
        ax.legend(); ax.grid(True)
        st.pyplot(fig)
        dxf = generate_dxf_smart((mt, md), mode="long")
        st.download_button("📥 DOWNLOAD DXF LONG", dxf, "Long_Result.dxf", "application/dxf")

# --- TAB 3: PETA SITUASI & GIS ---
with tab3:
    st.header("Peta Situasi & Data Generator")
    
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        st.info("Input DEM & Trase untuk melihat Peta Situasi.")
        f_dem = st.file_uploader("1. DEM (.tif)", type=["tif", "tiff"])
        f_shp = st.file_uploader("2. Trase (.geojson)", type=["json", "geojson"])
        
        st.markdown("---")
        mode_gen = st.selectbox("Mode Output:", ["Cross Section Data", "Long Section Data"])
        interval = st.number_input("Interval (m):", 50)
        wl, wr = 20, 20
        if mode_gen == "Cross Section Data":
            wl = st.number_input("Lebar Kiri (m):", 20)
            wr = st.number_input("Lebar Kanan (m):", 20)

    with col_g2:
        if f_dem and f_shp:
            try:
                # RESET POINTER UNTUK BACA ULANG
                f_dem.seek(0)
                f_shp.seek(0)
                
                with rasterio.open(f_dem) as src:
                    gdf = gpd.read_file(f_shp)
                    if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # 1. Peta Kontur (Image)
                    data = src.read(1)
                    data = np.ma.masked_where(data == src.nodata, data)
                    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                    im = ax.imshow(data, extent=extent, cmap='terrain', alpha=0.8)
                    fig.colorbar(im, label='Elevasi (m)', shrink=0.5)
                    
                    # 2. Trase
                    gdf.plot(ax=ax, color='red', linewidth=2, label='Trase')
                    
                    # 3. Label STA
                    line = gdf.geometry.iloc[0]
                    stations = np.arange(0, line.length, interval)
                    for dist in stations:
                        pt = line.interpolate(dist)
                        ax.annotate(f"STA {int(dist)}", (pt.x, pt.y), color='black', 
                                    fontsize=8, ha='center', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
                    
                    # 4. Grid
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    ax.set_title("Peta Situasi & Grid Koordinat")
                    st.pyplot(fig)
                    
                    st.markdown("### 📥 Download Data")
                    if st.button("GENERATE DATA (.CSV)"):
                        f_dem.seek(0); f_shp.seek(0)
                        if mode_gen == "Cross Section Data":
                            res, err = generate_cross_from_gis(f_dem, f_shp, interval, wl, wr)
                            if not err:
                                st.download_button("Download CSV PCLP", pclp_data_to_csv(res), "GIS_Cross.csv", "text/csv")
                        else:
                            res, err = generate_cross_from_gis(f_dem, f_shp, interval, 0, 0)
                            if not err:
                                pts = [[float(i)*interval, item['points'][0][1]] for i, item in enumerate(res) if item['points']]
                                df_l = pd.DataFrame(pts, columns=["Jarak", "Elevasi"])
                                st.download_button("Download CSV Long", df_l.to_csv(index=False).encode(), "GIS_Long.csv", "text/csv")
            except Exception as e: st.error(f"Gagal render peta: {e}")
