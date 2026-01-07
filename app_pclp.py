import ezdxf
from shapely.geometry import Polygon
import pandas as pd

def hitung_cut_fill(data_tanah_asli, data_desain_saluran):
    """
    Menghitung luas Galian (Cut) dan Timbunan (Fill) menggunakan Shapely.
    """
    # Tentukan datum bantu (dasar poligon) agar area tertutup
    # Ambil nilai Y minimum dari seluruh data dikurangi margin aman
    min_y = min([p[1] for p in data_tanah_asli] + [p[1] for p in data_desain_saluran])
    datum = min_y - 10.0 

    # 1. Buat Poligon Tanah Asli (Closed Polygon)
    # Urutan: Titik Tanah -> Turun ke Datum Kanan -> Geser ke Datum Kiri -> Naik ke Awal
    tanah_poly_points = data_tanah_asli + [
        (data_tanah_asli[-1][0], datum), 
        (data_tanah_asli[0][0], datum)
    ]
    poly_tanah = Polygon(tanah_poly_points)
    
    # 2. Buat Poligon Desain (Closed Polygon)
    desain_poly_points = data_desain_saluran + [
        (data_desain_saluran[-1][0], datum), 
        (data_desain_saluran[0][0], datum)
    ]
    poly_desain = Polygon(desain_poly_points)

    # Validasi Geometri
    if not poly_tanah.is_valid: poly_tanah = poly_tanah.buffer(0)
    if not poly_desain.is_valid: poly_desain = poly_desain.buffer(0)
    
    # 3. Hitung Boolean Operations
    try:
        # Area Desain yang berada DI BAWAH Tanah Asli = Galian (Intersection)
        area_cut = poly_desain.intersection(poly_tanah).area
        
        # Area Desain yang berada DI ATAS Tanah Asli = Timbunan (Difference)
        # Logika: Area Desain Total - Area yang beririsan dengan tanah
        area_fill = poly_desain.difference(poly_tanah).area
        
    except Exception as e:
        print(f"Error Geometri: {e}")
        area_cut, area_fill = 0, 0

    return area_cut, area_fill

def buat_gambar_dxf(nama_file, tanah_points, desain_points, cut_val, fill_val):
    """
    Generate file DXF dengan layer standar.
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Setup Layer
    if 'TANAH_ASLI' not in doc.layers:
        doc.layers.add(name='TANAH_ASLI', color=8) # Abu-abu
    if 'DESAIN_SALURAN' not in doc.layers:
        doc.layers.add(name='DESAIN_SALURAN', color=1) # Merah (Cut)
    
    # Gambar Garis
    msp.add_lwpolyline(tanah_points, dxfattribs={'layer': 'TANAH_ASLI'})
    msp.add_lwpolyline(desain_points, dxfattribs={'layer': 'DESAIN_SALURAN'})

    # Tambahkan Label Text
    teks_info = f"Cut: {cut_val:.3f} m2 | Fill: {fill_val:.3f} m2"
    
    # Posisi teks di atas tengah gambar
    center_x = (tanah_points[0][0] + tanah_points[-1][0]) / 2
    max_y = max([p[1] for p in tanah_points])
    
    msp.add_text(teks_info, dxfattribs={'height': 0.3}).set_placement(
        (center_x, max_y + 1.0), 
        align=ezdxf.const.TEXT_ALIGN_CENTER
    )

    doc.saveas(nama_file)
    print(f"[SUKSES] File '{nama_file}' berhasil dibuat.")

# --- MAIN PROGRAM (SIMULASI) ---
if __name__ == "__main__":
    # 1. Data Dummy (Koordinat X, Y)
    # Tanah: Cekung di tengah
    raw_tanah = [(-5, 12.0), (-2, 10.5), (0, 10.0), (2, 10.5), (5, 11.0)] 
    
    # Desain: Saluran Trapesium (Bottom=2m, Tinggi=2m, Talud 1:1)
    # Elevasi dasar rencana = 9.0
    # Koordinat: (-3, 11), (-1, 9), (1, 9), (3, 11)
    raw_desain = [(-3, 11.0), (-1, 9.0), (1, 9.0), (3, 11.0)]

    # 2. Proses Hitungan
    cut, fill = hitung_cut_fill(raw_tanah, raw_desain)
    
    print(f"Hasil Perhitungan:")
    print(f"Luas Galian (Cut)   : {cut:.3f} m2")
    print(f"Luas Timbunan (Fill): {fill:.3f} m2")

    # 3. Generate Gambar
    buat_gambar_dxf("Output_Cross_Section.dxf", raw_tanah, raw_desain, cut, fill)
