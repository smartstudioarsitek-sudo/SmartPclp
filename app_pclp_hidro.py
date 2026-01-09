import streamlit as st
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
import geopandas as gpd
import fiona
import tempfile
import os

# --- Pastikan kode import di atas ada di paling atas file ---

# [BAGIAN KODE YANG ERROR SEBELUMNYA SUDAH DIPERBAIKI DI BAWAH INI]

# TAMPILKAN PETA
if 'active_dem' in st.session_state:
    st.subheader("2. Peta Interaktif")
    
    # UPDATE: Set default ke Satellite Hybrid agar sungai terlihat jelas
    m = leafmap.Map(google_map="HYBRID") 
    
    # Coba visualisasi DEM (Jika ringan akan muncul, jika berat di-skip)
    try:
        # Cek apakah active_dem valid sebelum di-add
        if st.session_state['active_dem'] is not None:
            m.add_raster(st.session_state['active_dem'], layer_name="Topografi (DEM)", colormap="terrain", opacity=0.6)
    except Exception as e:
        # Jika gagal visualisasi, beri info tapi JANGAN STOP program
        st.caption(f"ℹ️ Visualisasi warna topografi dinonaktifkan demi performa: {e}")

    # Tampilkan Batas Wilayah (Jika ada variabel up_boundary dari upload sebelumnya)
    # Kita gunakan try-except untuk safety jika variabel up_boundary belum terdefinisi
    try:
        if 'up_boundary' in locals() and up_boundary is not None:
            fiona.drvsupport.supported_drivers['KML'] = 'rw'
            fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
            
            # Simpan file sementara untuk dibaca geopandas
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_boundary.name.split('.')[-1]}") as tmp:
                tmp.write(up_boundary.getbuffer())
                tmp_path = tmp.name
            
            gdf = gpd.read_file(tmp_path)
            bounds = gdf.total_bounds
            m.zoom_to_bounds((bounds[0], bounds[1], bounds[2], bounds[3]))
            
            # Tampilkan garis batas dengan warna cerah (Cyan)
            style = {'fillColor': '#00000000', 'color': 'cyan', 'weight': 3}
            m.add_gdf(gdf, layer_name="Batas Wilayah", style=style)
    except Exception as e:
        # Abaikan jika gagal load batas wilayah agar peta tetap muncul
        pass
        
    # Render Peta ke Streamlit
    map_out = st_folium(m, height=500, width=None)
    
    # LOGIKA KLIK (Menghitung DAS)
    if map_out and map_out['last_clicked']:
        lat = map_out['last_clicked']['lat']
        lng = map_out['last_clicked']['lng']
        st.info(f"📍 Koordinat Klik: {lat:.5f}, {lng:.5f}")
        
        if 'engine' in st.session_state:
            with st.spinner("⏳ Sedang menghitung batas DAS..."):
                try:
                    eng = st.session_state['engine']
                    # Jalankan delineasi
                    poly_das = eng.delineate(lng, lat)
                    
                    if poly_das:
                        # Estimasi kasar area (Konversi derajat ke km persegi di ekuator)
                        area_km2 = (poly_das.area * 111.32 * 111.32) 
                        st.success(f"✅ DAS Berhasil Dibuat! Luas: ±{area_km2:.2f} km²")
                        
                        # Siapkan Download GeoJSON
                        gdf_res = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                        st.download_button(
                            label="📥 Download GeoJSON DAS",
                            data=gdf_res.to_json(),
                            file_name="hasil_das.geojson",
                            mime="application/json"
                        )
                    else:
                        st.warning("⚠️ Titik klik di luar alur sungai/DEM. Coba klik pas di tengah lembah pada peta.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat delineasi: {e}")
