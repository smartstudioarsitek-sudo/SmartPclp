import streamlit as st
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, LineString, Point
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patheffects as pe
import io
import numpy as np
import math

# Cek Library Geospasial (Optional untuk Tab 4)
try:
    import geopandas as gpd
    import rasterio
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False

# ==========================================
# 1. PARSER ENGINE
# ==========================================
def parse_pclp_smart(df):
    """Parser Mode Blok Horizontal."""
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
    """Parser Mode Tabel Panjang."""
    header_idx = -1
    col_dist_idx = -1
    col_elev_idx = -1
    
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
    res = parse_pclp_smart(df)
    if not res: res = parse_long_tabular(df)
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
# 2. SITUASI & GIS ENGINE
# ==========================================
def calculate_situasi_from_excel(plan_df, ogl_data):
    """Mengubah Data Plan + OGL menjadi Koordinat Global (X, Y, Z) untuk Kontur."""
    # 1. Baca Data Plan (Cari kolom X, Y, Patok)
    header_idx = -1
    col_map = {'X': -1, 'Y': -1, 'Patok': -1}
    
    for r in range(min(15, len(plan_df))):
        row_vals = [str(v).strip().upper() for v in plan_df.iloc[r].values]
        if 'X' in row_vals and 'Y' in row_vals:
            header_idx = r
            for c, val in enumerate(row_vals):
                if val == 'X': col_map['X'] = c
                elif val == 'Y': col_map['Y'] = c
                elif 'PATOK' in val or 'NAME' in val or 'NO.' in val: col_map['Patok'] = c
            break
            
    if header_idx == -1: return None, "Header 'X', 'Y', 'Patok' tidak ditemukan di sheet Plan."
    
    plan_pts = []
    for i in range(header_idx+1, len(plan_df)):
        try:
            x = float(plan_df.iloc[i, col_map['X']])
            y = float(plan_df.iloc[i, col_map['Y']])
            # Coba ambil nama patok
            name = str(plan_df.iloc[i, col_map['Patok']]) if col_map['Patok'] != -1 else f"P{i}"
            if not pd.isna(x) and not pd.isna(y):
                plan_pts.append({'x': x, 'y': y, 'name': name})
        except: continue
        
    if not plan_pts: return None, "Data Koordinat Plan kosong."

    # 2. Gabungkan dengan OGL (Transformasi Lokal ke Global)
    xyz_cloud = []
    limit = min(len(plan_pts), len(ogl_data))
    
    for i in range(limit):
        center = plan_pts[i]
        ogl = ogl_data[i]['points']
        
        # Hitung Arah Garis (Azimuth)
        if i < len(plan_pts) - 1:
            dx = plan_pts[i+1]['x'] - center['x']
            dy = plan_pts[i+1]['y'] - center['y']
        elif i > 0:
            dx = center['x'] - plan_pts[i-1]['x']
            dy = center['y'] - plan_pts[i-1]['y']
        else:
            dx, dy = 1, 0 # Default

        vec_len = math.sqrt(dx*dx + dy*dy)
        if vec_len == 0: continue
        
        # Vektor Normal (Tegak Lurus Trase)
        nx = -dy / vec_len
        ny = dx / vec_len
        
        # Transformasi
        for pt in ogl:
            offset = pt[0] # Jarak dari as
            elev = pt[1]
            
            gx = center['x'] + (nx * offset)
            gy = center['y'] + (ny * offset)
            xyz_cloud.append([gx, gy, elev])
            
    return {'trase': plan_pts, 'xyz': xyz_cloud}, None

def sample_dem_at_point(src, x, y):
    try:
        vals = list(src.sample([(x, y)]))
        val = vals[0][0]
        if val == src.nodata: return None
        return float(val)
    except: return None

def generate_cross_from_gis(dem_file, align_file, interval, left_w, right_w):
    if not HAS_GEO_LIBS: return None, "Library Missing."
    try:
        gdf = gpd.read_file(align_file)
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
                    if elev is not None: points.append((off, elev))
                if points: pclp_data.append({'STA': f"STA {int(dist)}", 'points': points})
        return pclp_data, None
    except Exception as e: return None, str(e)

def pclp_data_to_csv(data_list):
    lines = ["Generated by PCLP Studio,,,,,,,,,"]
    for i, item in enumerate(data_list):
        row_x = ["", "", "Elv. Min", "X"] + [f"{x:.2f}" for x in [p[0] for p in item['points']]]
        lines.append(",".join(row_x))
        row_y = [i+1, item['STA'], min([p[1] for p in item['points']]), "Y"] + [f"{y:.3f}" for y in [p[1] for p in item['points']]]
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
            if t_pts: msp.add_lwpolyline([(p[0]+offset[0], p[1]+offset[1]) for p in t_pts], dxfattribs={'layer': 'TANAH'})
            if d_pts: msp.add_lwpolyline([(p[0]+offset[0], p[1]+offset[1]) for p in d_pts], dxfattribs={'layer': 'DESAIN'})
            info = f"{item['STA']}\\P"
            if t_pts and d_pts: info += f"Cut: {item['cut']:.2f} m2\\PFill: {item['fill']:.2f} m2"
            cy = max([p[1] for p in t_pts] if t_pts else [0])
            cx = (t_pts[0][0] if t_pts else 0) + offset[0]
            msp.add_mtext(info, dxfattribs={'char_height':0.4, 'layer':'TEXT'}).set_location(insert=(cx, cy+4), attachment_point=ezdxf.const.MTEXT_TOP_CENTER)
            count += 1
            
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode('utf-8')

# ==========================================
# 4. UI STREAMLIT
# ==========================================
st.set_page_config(page_title="PCLP Studio v5.2", layout="wide")
st.title("🚜 PCLP Studio v5.2 (Anti-Crash)")

tab1, tab2, tab3, tab4 = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🗺️ PETA SITUASI (EXCEL)", "🌍 GIS (DEM)"])

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
                    
                    # RESET STATE AGAR TIDAK ERROR INDEX
                    st.session_state['res_cross'] = final_res
                    st.session_state['sta_index'] = 0 
                    st.success("Data Terbaca!")
            except Exception as e: st.error(f"Error: {e}")

    with col2:
        if 'res_cross' in st.session_state:
            data = st.session_state['res_cross']
            
            # --- NAVIGASI SAFE MODE (ANTI-CRASH) ---
            st.subheader("2. Visualisasi")
            c_nav1, c_nav2, c_nav3 = st.columns([1, 2, 1])
            
            # Logic Tombol
            if c_nav1.button("◀ Prev"): st.session_state['sta_index'] = max(0, st.session_state['sta_index'] - 1)
            if c_nav3.button("Next ▶"): st.session_state['sta_index'] = min(len(data) - 1, st.session_state['sta_index'] + 1)
            
            # Logic Slider (Dengan Safety Check)
            current_idx = st.session_state.get('sta_index', 0)
            if current_idx >= len(data): current_idx = 0 # Safety if data changed
            
            sta_options = [d['STA'] for d in data]
            selected_sta = c_nav2.selectbox("Pilih STA:", sta_options, index=current_idx)
            
            # Sync back if user changed slider
            if selected_sta in sta_options:
                new_idx = sta_options.index(selected_sta)
                if new_idx != st.session_state['sta_index']:
                    st.session_state['sta_index'] = new_idx
            
            item = data[st.session_state['sta_index']]
            
            # Offset Adjuster
            c_adj1, c_adj2 = st.columns(2)
            dx = c_adj1.number_input("Geser Tanah X (m):", 0.0, step=0.5, key='dx')
            dy = c_adj2.number_input("Geser Tanah Y (m):", 0.0, step=0.5, key='dy')
            
            t_pts = [(p[0]+dx, p[1]+dy) for p in item['points_tanah']]
            d_pts = item['points_desain']
            cut, fill = hitung_cut_fill(t_pts, d_pts)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            if t_pts: ax.plot(*zip(*t_pts), 'k-o', markersize=4, label='Tanah')
            if d_pts: ax.plot(*zip(*d_pts), 'r-o', markersize=4, label='Desain')
            ax.set_title(f"{item['STA']} | Cut: {cut:.2f} | Fill: {fill:.2f}")
            ax.legend(); ax.grid(True, linestyle=':'); ax.set_aspect('equal')
            st.pyplot(fig)
            
            dxf_bytes = generate_dxf_smart(data, mode="cross")
            st.download_button("📥 DOWNLOAD DXF (ALL STA)", dxf_bytes, "Cross_Result.dxf", "application/dxf")

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

# --- TAB 3: PETA SITUASI (EXCEL) ---
with tab3:
    st.header("Peta Situasi & Kontur (Dari Excel)")
    

[Image of topographic map with contour lines]

    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.info("Upload File Excel PCLP yang berisi Sheet 'DataPlan' (Trase) dan 'DataOGL' (Cross).")
        f_sit = st.file_uploader("Upload Excel:", key="usit")
        if f_sit:
            xls_s = pd.ExcelFile(f_sit)
            s_plan = st.selectbox("Sheet Plan (Trase):", xls_s.sheet_names, index=min(3, len(xls_s.sheet_names)-1))
            s_ogl_sit = st.selectbox("Sheet OGL (Cross):", xls_s.sheet_names, index=1)
            
            if st.button("GENERATE PETA"):
                df_plan = pd.read_excel(f_sit, sheet_name=s_plan, header=None)
                df_ogl_sit = pd.read_excel(f_sit, sheet_name=s_ogl_sit, header=None)
                
                ogl_parsed = combined_parser(df_ogl_sit)
                res_sit, err = calculate_situasi_from_excel(df_plan, ogl_parsed)
                
                if err: st.error(err)
                else: st.session_state['res_situasi'] = res_sit

    with col_s2:
        if 'res_situasi' in st.session_state:
            data = st.session_state['res_situasi']
            trase = data['trase']
            xyz = data['xyz']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 1. Kontur
            if xyz:
                x = [p[0] for p in xyz]
                y = [p[1] for p in xyz]
                z = [p[2] for p in xyz]
                triang = mtri.Triangulation(x, y)
                contour = ax.tricontourf(triang, z, levels=15, cmap='terrain')
                fig.colorbar(contour, label='Elevasi (m)')
            
            # 2. Trase
            tx = [p['x'] for p in trase]
            ty = [p['y'] for p in trase]
            ax.plot(tx, ty, 'r-', linewidth=2, label='Trase')
            
            # 3. Label
            for p in trase:
                ax.annotate(p['name'], (p['x'], p['y']), fontsize=8, 
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            
            ax.grid(True, which='both')
            ax.set_aspect('equal')
            ax.set_title("Peta Situasi & Kontur")
            st.pyplot(fig)

# --- TAB 4: GIS (DEM) ---
with tab4:
    st.header("GIS Tools (Global Mapper DEM)")
    if HAS_GEO_LIBS:
        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            f_dem = st.file_uploader("1. DEM (.tif)", type=["tif"])
            f_shp = st.file_uploader("2. Trase (.geojson)", type=["json", "geojson"])
            mode_gen = st.selectbox("Output:", ["Cross Section Data", "Long Section Data"])
            interval = st.number_input("Interval (m):", 50)
            wl, wr = 20, 20
            if mode_gen == "Cross Section Data":
                wl = st.number_input("Lebar Kiri:", 20)
                wr = st.number_input("Lebar Kanan:", 20)
            
        with col_g2:
            if f_dem and f_shp and st.button("GENERATE DATA"):
                f_dem.seek(0); f_shp.seek(0)
                if mode_gen == "Cross Section Data":
                    res, err = generate_cross_from_gis(f_dem, f_shp, interval, wl, wr)
                    if not err:
                        st.download_button("Download CSV", pclp_data_to_csv(res), "GIS_Cross.csv", "text/csv")
                else:
                    res, err = generate_cross_from_gis(f_dem, f_shp, interval, 0, 0)
                    if not err:
                        pts = [[float(i)*interval, item['points'][0][1]] for i, item in enumerate(res) if item['points']]
                        st.download_button("Download CSV", pd.DataFrame(pts).to_csv(index=False).encode(), "GIS_Long.csv", "text/csv")
    else:
        st.warning("Install `geopandas rasterio` untuk fitur ini.")
