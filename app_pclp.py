import streamlit as st
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import io

# --- 1. Fungsi Hitungan (Engine) ---
def hitung_cut_fill(tanah, desain):
    # Buat datum
    all_y = [p[1] for p in tanah] + [p[1] for p in desain]
    datum = min(all_y) - 5.0

    # Buat Poligon
    poly_tanah = Polygon(tanah + [(tanah[-1][0], datum), (tanah[0][0], datum)])
    poly_desain = Polygon(desain + [(desain[-1][0], datum), (desain[0][0], datum)])

    if not poly_tanah.is_valid: poly_tanah = poly_tanah.buffer(0)
    if not poly_desain.is_valid: poly_desain = poly_desain.buffer(0)

    try:
        area_cut = poly_desain.intersection(poly_tanah).area
        area_fill = poly_desain.difference(poly_tanah).area
    except:
        area_cut, area_fill = 0, 0
    
    return area_cut, area_fill

# --- 2. Fungsi Generate DXF ke Memory ---
def get_dxf_binary(tanah, desain, cut, fill):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Layer
    doc.layers.add(name='TANAH_ASLI', color=8)
    doc.layers.add(name='DESAIN_SALURAN', color=1)
    
    # Gambar
    msp.add_lwpolyline(tanah, dxfattribs={'layer': 'TANAH_ASLI'})
    msp.add_lwpolyline(desain, dxfattribs={'layer': 'DESAIN_SALURAN'})
    
    # Teks
    info = f"Cut: {cut:.2f} m2 | Fill: {fill:.2f} m2"
    cx = (tanah[0][0] + tanah[-1][0]) / 2
    my = max([p[1] for p in tanah])
    
    text = msp.add_text(info, dxfattribs={'height': 0.3})
    text.set_placement((cx, my + 1.5), align=TextEntityAlignment.CENTER)
    
    # Simpan ke Buffer (Bukan file fisik) agar bisa didownload tombol
    output = io.StringIO()
    doc.write(output)
    return output.getvalue().encode('utf-8')

# --- 3. Tampilan Web (Streamlit) ---
st.set_page_config(page_title="PCLP Modern", layout="wide")
st.title("🚜 PCLP Modern: Cut & Fill Calculator")

# Kolom Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Input Tanah Asli")
    # Default data dummy
    raw_tanah_str = st.text_area("Koordinat Tanah (x,y)", 
                                 "-5,12\n-2,10.5\n0,10\n2,10.5\n5,11")
    
with col2:
    st.subheader("2. Input Desain")
    raw_desain_str = st.text_area("Koordinat Desain (x,y)", 
                                  "-3,11\n-1,9\n1,9\n3,11")

# Tombol Proses
if st.button("Hitung & Gambar"):
    try:
        # Parsing Text ke List of Tuples
        tanah_pts = [tuple(map(float, line.split(','))) for line in raw_tanah_str.split('\n') if line]
        desain_pts = [tuple(map(float, line.split(','))) for line in raw_desain_str.split('\n') if line]

        # Hitung
        cut, fill = hitung_cut_fill(tanah_pts, desain_pts)

        # Tampilkan Hasil Angka
        st.success(f"✅ Selesai! Cut: **{cut:.3f} m²** | Fill: **{fill:.3f} m²**")

        # --- PREVIEW GAMBAR (Matplotlib) ---
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot Tanah
        t_x, t_y = zip(*tanah_pts)
        ax.plot(t_x, t_y, label='Tanah Asli', color='gray', linewidth=2, linestyle='--')
        ax.fill_between(t_x, t_y, min(t_y)-5, color='gray', alpha=0.1)

        # Plot Desain
        d_x, d_y = zip(*desain_pts)
        ax.plot(d_x, d_y, label='Desain Rencana', color='red', linewidth=2)
        
        ax.set_title(f"Cross Section Preview (Cut: {cut:.2f} | Fill: {fill:.2f})")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')
        
        # Tampilkan di Streamlit
        st.pyplot(fig)

        # --- DOWNLOAD DXF ---
        dxf_data = get_dxf_binary(tanah_pts, desain_pts, cut, fill)
        st.download_button(
            label="📥 Download File AutoCAD (.dxf)",
            data=dxf_data,
            file_name="cross_section.dxf",
            mime="application/dxf"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan format data: {e}")