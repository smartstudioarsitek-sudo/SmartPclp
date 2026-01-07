import streamlit as st
import pandas as pd
import ezdxf
# Import Enum untuk kompatibilitas Text Alignment terbaru
from ezdxf.enums import TextEntityAlignment 
from shapely.geometry import Polygon, LineString, Point
import matplotlib.pyplot as plt
import io
import numpy as np
import math

# Cek Library Geospasial (Optional)
try:
    import geopandas as gpd
    import rasterio
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False

# ==========================================
# 1. CORE ENGINE: PARSER & MATH
# ==========================================

def parse_pclp_smart(df):
    """Parser universal PCLP Horizontal."""
    parsed_data = []
    i = 0
    # Loop baris demi baris
    while i < len(df):
        row = df.iloc[i]
        x_col_idx = -1
        
        # Scan 'X' (Cek 20 kolom pertama)
        max_col = min(20, len(row))
        for c in range(max_col):
            try:
                if str(row[c]).strip().upper() == 'X':
                    x_col_idx = c
                    break
            except: continue
        
        # Cek 'Y' di bawahnya
        if x_col_idx != -1 and (i + 1 < len(df)):
            try:
                val_y = str(df.iloc[i+1][x_col_idx]).strip().upper()
                if val_y == 'Y':
                    # --- FOUND BLOCK ---
                    sta_name = f"STA_{i}"
                    if x_col_idx >= 2:
                        val_sta = str(df.iloc[i+1][x_col_idx - 2]).strip()
                        if val_sta and val_sta.lower() != 'nan': sta_name = val_sta
                    if sta_name.endswith('.0'): sta_name = sta_name[:-2]

                    # Ambil Data
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
                        # Sort points by X (PENTING untuk Long Section)
                        points.sort(key=lambda p: p[0])
                        parsed_data.append({'STA': sta_name, 'points': points})
                    
                    i += 1 
            except: pass
        i += 1
    return parsed_data

def hitung_cut_fill(tanah_pts, desain_pts):
    """Menghitung luas. Return 0 jika salah satu data kosong."""
    if not tanah_pts or not desain_pts: return 0, 0
    
    # Datum bantu
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
# 2. GIS ENGINE (GENERATE FROM DEM)
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
        if gdf.crs.is_geographic: gdf = gdf.to_crs(epsg=3857)
        
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
    """Convert list ke CSV format PCLP."""
    lines = []
    lines.append("Generated by PCLP Studio,,,,,,,,,")
    for i, item in enumerate(data_list):
        sta = item['STA']
        pts = item['points']
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        row_x = ["", "", "Elv. Min", "X"] + [f"{x:.2f}" for x in xs]
        lines.append(",".join(row_x))
        row_y = [i+1, sta, min(ys), "Y"] + [f"{y:.3f}" for y in ys]
        lines.append(",".join([str(x) for x in row_y]))
        lines.append(",,,,,,,,,")
    return "\n".join(lines)

# ==========================================
# 3. DXF GENERATORS (FIXED FOR EZDXF V1.0+)
# ==========================================
def generate_dxf_smart(results, mode="cross"):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Layers
    if 'TANAH' not in doc.layers: doc.layers.add(name='TANAH', color=8)
    if 'DESAIN' not in doc.layers: doc.layers.add(name='DESAIN', color=1)
    if 'TEXT' not in doc.layers: doc.layers.add(name='TEXT', color=7)
    
    if mode == "long":
        # LONG SECTION
        tanah, desain = results
        if tanah: msp.add_lwpolyline(tanah, dxfattribs={'layer': 'TANAH'})
        if desain: msp.add_lwpolyline(desain, dxfattribs={'layer': 'DESAIN'})
        
        start_x = tanah[0][0] if tanah else 0
        
        # Gunakan MText yang aman
        msp.add_mtext("LONG SECTION PROFILE", dxfattribs={'char_height': 2.0, 'layer': 'TEXT'}).set_location(
            insert=(start_x, 10), 
            attachment_point=ezdxf.const.MTEXT_TOP_LEFT
        )
    else:
        # CROSS SECTION
        x_gap = 60
        count = 0
        for item in results:
            offset = (count * x_gap, 0)
            t_pts = item.get('points_tanah', [])
            d_pts = item.get('points_desain', [])
            
            # Gambar
            if t_pts:
                draw_t = [(p[0]+offset[0], p[1]+offset[1]) for p in t_pts]
                msp.add_lwpolyline(draw_t, dxfattribs={'layer': 'TANAH'})
            if d_pts:
                draw_d = [(p[0]+offset[0], p[1]+offset[1]) for p in d_pts]
                msp.add_lwpolyline(draw_d, dxfattribs={'layer': 'DESAIN'})
            
            # Teks Info
            info = f"{item['STA']}\\P"
            if t_pts and d_pts:
                info += f"Cut: {item['cut']:.2f} m2\\PFill: {item['fill']:.2f} m2"
            else: 
                info += "Data Parsial"
                
            # Hitung Center untuk teks
            cy = max([p[1] for p in t_pts] if t_pts else [0])
            cx = (draw_t[0][0] if t_pts else 0)
            
            # MText Modern (Aman untuk ezdxf v1.0+)
            msp.add_mtext(info, dxfattribs={'char_height':0.4, 'layer':'TEXT'}).set_location(
                insert=(cx, cy+4), 
                attachment_point=ezdxf.const.MTEXT_TOP_CENTER
            )
            count += 1
            
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

# ==========================================
# 4. UI STREAMLIT
# ==========================================
st.set_page_config(page_title="PCLP Studio v3.1", layout="wide")
st.title("🚜 PCLP Studio v3.1 (Safe Mode)")

if not HAS_GEO_LIBS:
    st.warning("⚠️ Fitur GIS non-aktif. Mohon install: `pip install geopandas rasterio`")

tab1, tab2, tab3 = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🌍 GENERATE DATA (GIS)"])

# --- TAB 1: CROSS SECTION ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Input Data")
        f_cross = st.file_uploader("Upload Excel Cross", key="uc")
        mode_match = st.radio("Mode Pencocokan:", ["Paksa Urutan (Index)", "Match Nama STA"], key="mm")
        
        if f_cross:
            # --- SAFE FILE LOADER ---
            try:
                xls = pd.ExcelFile(f_cross)
                valid_file = True
            except ValueError:
                st.error("❌ Format File Salah! Harap 'Save As' menjadi **.xlsx**.")
                valid_file = False
            except ImportError:
                st.error("❌ Library xlrd kurang. Update requirements.txt.")
                valid_file = False
            
            if valid_file:
                s_ogl = st.selectbox("Sheet Tanah:", ["(Tidak Ada)"] + xls.sheet_names, index=1)
                s_des = st.selectbox("Sheet Desain:", ["(Tidak Ada)"] + xls.sheet_names, index=min(2, len(xls.sheet_names)))
                
                if st.button("RUN CROSS SECTION"):
                    d_ogl, d_des = [], []
                    if s_ogl != "(Tidak Ada)":
                        d_ogl = parse_pclp_smart(pd.read_excel(f_cross, sheet_name=s_ogl, header=None))
                    if s_des != "(Tidak Ada)":
                        d_des = parse_pclp_smart(pd.read_excel(f_cross, sheet_name=s_des, header=None))
                    
                    final_res = []
                    # Handle jika salah satu kosong
                    if not d_ogl: d_ogl = [{'STA': f"Dummy_{i}", 'points': []} for i in range(len(d_des))]
                    if not d_des: d_des = [{'STA': f"Dummy_{i}", 'points': []} for i in range(len(d_ogl))]
                    
                    limit = max(len(d_ogl), len(d_des))
                    for i in range(limit):
                        t = d_ogl[i] if i < len(d_ogl) else {'STA': 'N/A', 'points': []}
                        d = d_des[i] if i < len(d_des) else {'STA': 'N/A', 'points': []}
                        sta = t['STA'] if t['points'] else d['STA']
                        c, f = hitung_cut_fill(t['points'], d['points'])
                        final_res.append({'STA': sta, 'cut': c, 'fill': f, 'points_tanah': t['points'], 'points_desain': d['points']})
                    
                    st.session_state['res_cross'] = final_res
                    st.success(f"Selesai! {len(final_res)} data diproses.")

    with col2:
        if 'res_cross' in st.session_state:
            res = st.session_state['res_cross']
            st.dataframe(pd.DataFrame(res)[['STA', 'cut', 'fill']], use_container_width=True, height=200)
            
            if res:
                item = res[0]
                fig, ax = plt.subplots(figsize=(10, 3))
                if item['points_tanah']:
                    tx, ty = zip(*item['points_tanah'])
                    ax.plot(tx, ty, 'k-o', markersize=3, label='Tanah')
                    ax.fill_between(tx, ty, min(ty)-2, color='gray', alpha=0.1)
                if item['points_desain']:
                    dx, dy = zip(*item['points_desain'])
                    ax.plot(dx, dy, 'r-o', markersize=3, label='Desain')
                ax.legend()
                ax.grid(True)
                ax.set_aspect('equal') # Agar proporsional
                st.pyplot(fig)
                
                dxf_bytes = generate_dxf_smart(res, mode="cross")
                st.download_button("📥 DOWNLOAD DXF", dxf_bytes, "Result_Cross.dxf", "application/dxf")

# --- TAB 2: LONG SECTION ---
with tab2:
    st.subheader("Long Section Profile")
    col1, col2 = st.columns([1, 2])
    with col1:
        f_long = st.file_uploader("Upload Excel Long", key="ul")
        if f_long:
            try:
                xls_l = pd.ExcelFile(f_long)
                sl_ogl = st.selectbox("Sheet Tanah:", ["(Tidak Ada)"] + xls_l.sheet_names, index=1, key="slo")
                sl_des = st.selectbox("Sheet Desain:", ["(Tidak Ada)"] + xls_l.sheet_names, index=min(2, len(xls_l.sheet_names)), key="sld")
                
                if st.button("RUN LONG SECTION"):
                    mt, md = [], []
                    if sl_ogl != "(Tidak Ada)":
                        for p in parse_pclp_smart(pd.read_excel(f_long, sheet_name=sl_ogl, header=None)): mt.extend(p['points'])
                        mt.sort(key=lambda x: x[0])
                    if sl_des != "(Tidak Ada)":
                        for p in parse_pclp_smart(pd.read_excel(f_long, sheet_name=sl_des, header=None)): md.extend(p['points'])
                        md.sort(key=lambda x: x[0])
                    st.session_state['res_long'] = (mt, md)
                    st.success("Berhasil digabung.")
            except ValueError: st.error("❌ Format File Salah (Gunakan .xlsx)")

    with col2:
        if 'res_long' in st.session_state:
            mt, md = st.session_state['res_long']
            fig, ax = plt.subplots(figsize=(10, 4))
            if mt: 
                x, y = zip(*mt)
                ax.plot(x, y, 'k-', linewidth=1, label='Tanah')
            if md: 
                x, y = zip(*md)
                ax.plot(x, y, 'r-', linewidth=2, label='Desain')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            dxf_bytes = generate_dxf_smart((mt, md), mode="long")
            st.download_button("📥 DOWNLOAD DXF", dxf_bytes, "Result_Long.dxf", "application/dxf")

# --- TAB 3: GENERATE DATA (GIS) ---
with tab3:
    st.header("Generate Data dari Global Mapper (DEM)")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        f_dem = st.file_uploader("1. Upload DEM (.tif)", type=["tif"])
        f_shp = st.file_uploader("2. Upload Trase (.geojson)", type=["json", "geojson"])
        mode_gen = st.selectbox("Output:", ["Cross Section Data", "Long Section Data"])
        interval = st.number_input("Interval (m):", 50)
        
        w_left, w_right = 0, 0
        if mode_gen == "Cross Section Data":
            c1, c2 = st.columns(2)
            w_left = c1.number_input("Lebar Kiri (m):", 20)
            w_right = c2.number_input("Lebar Kanan (m):", 20)
        
    with col_g2:
        if f_dem and f_shp and st.button("GENERATE DATA"):
            if mode_gen == "Cross Section Data":
                res_gis, err = generate_cross_from_gis(f_dem, f_shp, interval, w_left, w_right)
                if not err:
                    csv_str = pclp_data_to_csv(res_gis)
                    st.download_button("📥 DOWNLOAD CSV (PCLP)", csv_str, "Generated_Cross.csv", "text/csv")
            else:
                res_gis, err = generate_cross_from_gis(f_dem, f_shp, interval, 0, 0)
                if not err:
                    long_pts = [[float(i)*interval, item['points'][0][1]] for i, item in enumerate(res_gis) if item['points']]
                    df_long = pd.DataFrame(long_pts, columns=["Jarak", "Elevasi"])
                    st.dataframe(df_long, height=150)
                    st.download_button("📥 DOWNLOAD CSV LONG", df_long.to_csv(index=False).encode(), "Generated_Long.csv", "text/csv")
