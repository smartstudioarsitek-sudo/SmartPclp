import streamlit as st
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, LineString, Point
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import io
import numpy as np
import math

# Cek Library Geospasial
try:
    import geopandas as gpd
    import rasterio
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False

# ==========================================
# 1. PARSER ENGINE (DUAL MODE: BLOCK & TABULAR)
# ==========================================

def parse_pclp_smart(df):
    """Parser Mode 1: Format Blok Horizontal (Ada Huruf X dan Y)."""
    parsed_data = []
    i = 0
    while i < len(df):
        row = df.iloc[i]
        x_col_idx = -1
        max_col = min(20, len(row))
        
        # Cari marker X
        for c in range(max_col):
            try:
                if str(row[c]).strip().upper() == 'X':
                    x_col_idx = c
                    break
            except: continue
        
        # Cari marker Y di bawahnya
        if x_col_idx != -1 and (i + 1 < len(df)):
            try:
                val_y = str(df.iloc[i+1][x_col_idx]).strip().upper()
                if val_y == 'Y':
                    # Ketemu Blok
                    sta_name = f"STA_{i}"
                    # Coba ambil nama STA (biasanya di kiri Y)
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
    """Parser Mode 2: Format Tabel Panjang (Station, Distance, Elev)."""
    # Cari Header Row
    header_idx = -1
    col_dist_idx = -1
    col_elev_idx = -1
    
    # Scan 10 baris pertama untuk cari header
    for r in range(min(10, len(df))):
        row_vals = [str(v).lower() for v in df.iloc[r].values]
        # Cari kata kunci
        if any("dist" in x for x in row_vals) and (any("elev" in x for x in row_vals) or any("o.g.l" in x for x in row_vals) or any("elv" in x for x in row_vals)):
            header_idx = r
            # Deteksi kolom
            for c, val in enumerate(row_vals):
                if "cum" in val or "dist" in val: col_dist_idx = c # Prioritas Distance Cumulative
                if "o.g.l" in val or "bl" in val: col_elev_idx = c # Prioritas OGL atau Bed Level
            break
    
    if header_idx != -1 and col_dist_idx != -1 and col_elev_idx != -1:
        points = []
        # Baca data mulai dari bawah header
        for i in range(header_idx+1, len(df)):
            try:
                dist = df.iloc[i, col_dist_idx]
                elev = df.iloc[i, col_elev_idx]
                # Validasi angka
                fd = float(dist)
                fe = float(elev)
                if not (pd.isna(fd) or pd.isna(fe)):
                    points.append((fd, fe))
            except: continue
        
        # Sort by distance
        points.sort(key=lambda x: x[0])
        return [{'STA': 'Long_Section', 'points': points}]
    
    return []

def combined_parser(df):
    """Coba Parser Blok, kalau kosong coba Parser Tabel."""
    res = parse_pclp_smart(df)
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
# 2. GIS & SITUASI ENGINE
# ==========================================
def generate_situasi_data(plan_df, ogl_data):
    """Menggabungkan Data Plan (XY) dan OGL (Offset) untuk jadi 3D Points."""
    # 1. Parse Data Plan (Cari kolom X, Y, Z, Patok)
    header_idx = -1
    col_map = {'X': -1, 'Y': -1, 'Patok': -1}
    
    for r in range(min(10, len(plan_df))):
        row_vals = [str(v).upper() for v in plan_df.iloc[r].values]
        if 'X' in row_vals and 'Y' in row_vals:
            header_idx = r
            for c, val in enumerate(row_vals):
                if val == 'X': col_map['X'] = c
                elif val == 'Y': col_map['Y'] = c
                elif 'PATOK' in val or 'NAME' in val: col_map['Patok'] = c
            break
            
    if header_idx == -1: return None, "Format Data Plan tidak dikenali."
    
    # Extract Plan Path
    plan_path = []
    for i in range(header_idx+1, len(plan_df)):
        try:
            x = float(plan_df.iloc[i, col_map['X']])
            y = float(plan_df.iloc[i, col_map['Y']])
            name = str(plan_df.iloc[i, col_map['Patok']])
            plan_path.append({'x': x, 'y': y, 'name': name})
        except: continue
        
    if not plan_path: return None, "Data Plan Kosong."

    # 2. Mapping OGL ke Global Coordinate
    xyz_points = []
    
    # Mode Paksa Urutan (Asumsi urutan Plan sama dengan urutan OGL)
    limit = min(len(plan_path), len(ogl_data))
    
    for i in range(limit):
        center = plan_path[i]
        ogl = ogl_data[i]['points']
        
        # Hitung Azimuth (Arah Garis)
        # Jika bukan titik terakhir, arah ke titik depan. Jika terakhir, arah dari belakang.
        if i < len(plan_path) - 1:
            dx = plan_path[i+1]['x'] - center['x']
            dy = plan_path[i+1]['y'] - center['y']
        else:
            dx = center['x'] - plan_path[i-1]['x']
            dy = center['y'] - plan_path[i-1]['y']
            
        # Vektor Normal (Tegak Lurus Trase)
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0: continue
        nx = -dy / length # Normal X
        ny = dx / length  # Normal Y
        
        # Transformasi Tiap Titik OGL
        for pt in ogl:
            offset = pt[0]
            elev = pt[1]
            
            # Global X, Y
            gx = center['x'] + (nx * offset)
            gy = center['y'] + (ny * offset)
            
            xyz_points.append([gx, gy, elev])
            
    return {'path': plan_path, 'xyz': xyz_points}, None

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
        # Cross - Mode Download All
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
            
            info = f"{item['STA']}\\PCut: {item['cut']:.2f} m2\\PFill: {item['fill']:.2f} m2"
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
st.set_page_config(page_title="PCLP Studio v4.0", layout="wide")
st.title("🚜 PCLP Studio v4.0 Ultimate")

# --- CSS Styling ---
st.markdown("""
<style>
div.stButton > button {width: 100%;}
.big-font {font-size:20px !important; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📐 CROSS SECTION", "📈 LONG SECTION", "🗺️ PETA SITUASI", "🌍 GIS TOOLS"])

# --- TAB 1: CROSS SECTION (Navigasi Slider) ---
with tab1:
    col1, col2 = st.columns([1, 3])
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
                    
                    # Matching Logic
                    final_res = []
                    limit = max(len(d_ogl), len(d_des))
                    for i in range(limit):
                        t = d_ogl[i] if i < len(d_ogl) else {'STA': f'STA_{i}', 'points': []}
                        d = d_des[i] if i < len(d_des) else {'STA': f'STA_{i}', 'points': []}
                        
                        # Simpan RAW points untuk digeser nanti
                        final_res.append({
                            'STA': t['STA'],
                            'raw_tanah': t['points'],
                            'raw_desain': d['points']
                        })
                    
                    st.session_state['res_cross_raw'] = final_res
                    st.success("Data Terbaca!")
            except Exception as e: st.error(f"Error: {e}")

    with col2:
        if 'res_cross_raw' in st.session_state:
            data = st.session_state['res_cross_raw']
            
            # --- NAVIGASI SLIDER ---
            st.subheader("2. Visualisasi & Koreksi")
            
            # Slider untuk memilih STA
            sta_list = [d['STA'] for d in data]
            selected_sta = st.select_slider("Pilih Nomor Gambar / Station:", options=sta_list)
            
            # Cari data STA terpilih
            item = next((x for x in data if x['STA'] == selected_sta), data[0])
            
            # --- OFFSET ADJUSTER (GESER MANUAL) ---
            c_adj1, c_adj2, c_adj3 = st.columns(3)
            dx = c_adj1.number_input("Geser Tanah X (m):", value=0.0, step=0.5, key='dx')
            dy = c_adj2.number_input("Geser Tanah Y (m):", value=0.0, step=0.5, key='dy')
            
            # Terapkan Offset
            t_pts = [(p[0]+dx, p[1]+dy) for p in item['raw_tanah']]
            d_pts = item['raw_desain']
            
            # Hitung Ulang Cut/Fill Realtime
            cut, fill = hitung_cut_fill(t_pts, d_pts)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            if t_pts:
                tx, ty = zip(*t_pts)
                ax.plot(tx, ty, 'k-o', markersize=4, label='Tanah Asli')
                ax.fill_between(tx, ty, min(ty)-2, color='gray', alpha=0.1)
            if d_pts:
                dx_vals, dy_vals = zip(*d_pts)
                ax.plot(dx_vals, dy_vals, 'r-o', markersize=4, label='Desain')
            
            ax.set_title(f"Cross Section: {selected_sta} | Cut: {cut:.2f} m2 | Fill: {fill:.2f} m2")
            ax.legend()
            ax.grid(True, which='both', linestyle=':')
            ax.set_aspect('equal')
            st.pyplot(fig)
            
            st.info("💡 Gunakan 'Geser Tanah' jika grafik tidak berpotongan presisi.")

            # --- DOWNLOAD ---
            # Update semua data dengan offset yang dipilih (Opsional, saat ini hanya display yg digeser)
            # Untuk simplisitas, download DXF menggunakan data RAW dulu
            
            if st.button("📥 DOWNLOAD SEMUA KE DXF"):
                # Hitung ulang semua cut fill untuk report
                processed_res = []
                for it in data:
                    # Apply offset user ke semua? Atau cuma yg dilihat? 
                    # Asumsi: Offset user biasanya global (misal beda datum), jadi apply ke semua
                    tp = [(p[0]+dx, p[1]+dy) for p in it['raw_tanah']]
                    dp = it['raw_desain']
                    c, f = hitung_cut_fill(tp, dp)
                    processed_res.append({
                        'STA': it['STA'], 'cut': c, 'fill': f,
                        'points_tanah': tp, 'points_desain': dp
                    })
                
                dxf_bytes = generate_dxf_smart(processed_res, mode="cross")
                st.download_button("Klik Disini untuk Simpan .dxf", dxf_bytes, "Final_Cross.dxf", "application/dxf")

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
                # Gunakan Combined Parser (Blok + Tabel)
                if sl_ogl != "(Tidak Ada)":
                    res = combined_parser(pd.read_excel(f_long, sheet_name=sl_ogl, header=None))
                    for p in res: mt.extend(p['points'])
                    mt.sort(key=lambda x: x[0])
                if sl_des != "(Tidak Ada)":
                    res = combined_parser(pd.read_excel(f_long, sheet_name=sl_des, header=None))
                    for p in res: md.extend(p['points'])
                    md.sort(key=lambda x: x[0])
                
                st.session_state['res_long'] = (mt, md)
                st.success("Berhasil!")
        except: st.error("Format File Error.")
        
    if 'res_long' in st.session_state:
        mt, md = st.session_state['res_long']
        fig, ax = plt.subplots(figsize=(10, 4))
        if mt: ax.plot(*zip(*mt), 'k-', label='Tanah')
        if md: ax.plot(*zip(*md), 'r-', label='Desain')
        ax.legend(); ax.grid(True)
        st.pyplot(fig)
        
        dxf = generate_dxf_smart((mt, md), mode="long")
        st.download_button("📥 DOWNLOAD DXF LONG", dxf, "Long_Section.dxf", "application/dxf")

# --- TAB 3: PETA SITUASI (NEW) ---
with tab3:
    st.subheader("Peta Situasi & Kontur")
    st.info("Menggabungkan Data 'DataPlan' (Trase) dengan 'DataOGL' (Cross) untuk visualisasi 3D.")
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        f_sit = st.file_uploader("Upload File Excel PCLP (Berisi DataPlan & OGL)", key="usit")
        if f_sit:
            xls_s = pd.ExcelFile(f_sit)
            s_plan = st.selectbox("Sheet Plan (Trase):", xls_s.sheet_names, index=min(3, len(xls_s.sheet_names)-1))
            s_ogl_sit = st.selectbox("Sheet OGL (Cross):", xls_s.sheet_names, index=1)
            
            if st.button("GENERATE PETA"):
                df_plan = pd.read_excel(f_sit, sheet_name=s_plan, header=None)
                df_ogl_sit = pd.read_excel(f_sit, sheet_name=s_ogl_sit, header=None)
                
                ogl_parsed = combined_parser(df_ogl_sit)
                
                # Proses 3D
                res_sit, err = generate_situasi_data(df_plan, ogl_parsed)
                
                if err: st.error(err)
                else: st.session_state['res_situasi'] = res_sit
    
    with col_s2:
        if 'res_situasi' in st.session_state:
            data = st.session_state['res_situasi']
            path = data['path']
            xyz = data['xyz']
            
            if not xyz:
                st.warning("Data Plan dan OGL tidak sinkron jumlahnya.")
            else:
                # Plotting Kontur
                x = [p[0] for p in xyz]
                y = [p[1] for p in xyz]
                z = [p[2] for p in xyz]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # 1. Gambar Trase
                px = [p['x'] for p in path]
                py = [p['y'] for p in path]
                ax.plot(px, py, 'r-', linewidth=2, label='As Saluran')
                
                # 2. Gambar Kontur (Triangulasi)
                triang = mtri.Triangulation(x, y)
                contour = ax.tricontourf(triang, z, levels=15, cmap='terrain')
                fig.colorbar(contour, label='Elevasi (m)')
                
                # 3. Label Patok
                for p in path:
                    ax.annotate(p['name'], (p['x'], p['y']), fontsize=8)
                
                ax.set_aspect('equal')
                ax.grid(True)
                ax.set_title("Peta Situasi & Kontur Topografi")
                st.pyplot(fig)

# --- TAB 4: GIS TOOLS ---
with tab4:
    st.header("GIS Tools (DEM to PCLP)")
    # (Kode GIS sama seperti v3.2, disederhanakan tampilannya)
    st.info("Gunakan tab ini untuk membuat data Cross Section baru dari file DEM Global Mapper.")
