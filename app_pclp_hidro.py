import streamlit as st
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
import geopandas as gpd
import fiona
import tempfile
import os
import rioxarray as rxr  # Pastikan library ini ada untuk membaca DEM

# --- 1. JUDUL & SETUP ---
st.title("💧 Analisis Hidrologi & Delineasi DAS")

# Inisialisasi Session State jika belum ada
if 'active_dem' not in st.session_state:
    st.session_state['active_dem'] = None
if 'engine' not in st.session_state:
    # Disini harusnya inisialisasi engine hidrologi kakak (misal HydroEngine)
    # Jika kakak punya class engine sendiri, pastikan di-init di sini
    pass 

# --- 2. BAGIAN UPLOAD FILE (Ini yang mungkin hilang tadi) ---
st.subheader("1. Input Data Topografi")

uploaded_dem = st.file_uploader("Upload File DEM (.tif)", type=["tif", "tiff"])
up_boundary = st.file_uploader("Upload Batas Wilayah (Opsional - .kml/.geojson)", type=["kml", "json", "geojson"])

# Proses File DEM jika diupload
if uploaded_dem is not None:
    try:
        # Simpan file sementara agar bisa dibaca library
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_dem.getbuffer())
            dem_path = tmp.name
        
        # Simpan path atau data ke session state
        # (Sesuaikan dengan cara kerja engine kakak sebelumnya)
        # Disini saya asumsikan kita simpan path-nya untuk leafmap
        st.session_state['active_dem'] = dem_path
        
        # --- PENTING: Inisialisasi Engine Kakak Disini ---
        # Contoh: st.session_state['engine'] = HydroEngine(dem_path)
        # Karena saya tidak punya kode engine kakak, saya skip baris ini.
        # Pastikan kakak menambahkan logika load engine di sini jika perlu.
        
        st.success(f"✅ File DEM berhasil dimuat: {uploaded_dem.name}")
        
    except Exception as e:
        st.error(f"Gagal memuat file DEM: {e}")


# --- 3. BAGIAN PETA INTERAKTIF (Kode yang sudah diperbaiki) ---
st.divider()

# Cek apakah DEM sudah ada di session state
if st.session_state['active_dem']:
    st.subheader("2. Peta Interaktif")
    
    # Setup Peta
    m = leafmap.Map(google_map="HYBRID") 
    
    # Visualisasi DEM
    try:
        if st.session_state['active_dem'] is not None:
            m.add_raster(st.session_state['active_dem'], layer_name="Topografi (DEM)", colormap="terrain", opacity=0.6)
    except Exception as e:
        st.caption(f"ℹ️ Visualisasi raster skip (performa): {e}")

    # Visualisasi Batas Wilayah (Jika ada)
    try:
        if up_boundary is not None:
            fiona.drvsupport.supported_drivers['KML'] = 'rw'
            fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_boundary.name.split('.')[-1]}") as tmp:
                tmp.write(up_boundary.getbuffer())
                tmp_path = tmp.name
            
            gdf = gpd.read_file(tmp_path)
            bounds = gdf.total_bounds
            m.zoom_to_bounds((bounds[0], bounds[1], bounds[2], bounds[3]))
            
            style = {'fillColor': '#00000000', 'color': 'cyan', 'weight': 3}
            m.add_gdf(gdf, layer_name="Batas Wilayah", style=style)
    except Exception as e:
        pass
        
    # Tampilkan Peta
    map_out = st_folium(m, height=500, width=None)
    
    # Logika Klik & Delineasi
    if map_out and map_out['last_clicked']:
        lat = map_out['last_clicked']['lat']
        lng = map_out['last_clicked']['lng']
        st.info(f"📍 Koordinat Klik: {lat:.5f}, {lng:.5f}")
        
        # Cek apakah engine sudah siap
        if 'engine' in st.session_state and st.session_state['engine'] is not None:
            with st.spinner("⏳ Sedang menghitung batas DAS..."):
                try:
                    eng = st.session_state['engine']
                    poly_das = eng.delineate(lng, lat)
                    
                    if poly_das:
                        area_km2 = (poly_das.area * 111.32 * 111.32) 
                        st.success(f"✅ DAS Berhasil Dibuat! Luas: ±{area_km2:.2f} km²")
                        
                        gdf_res = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                        st.download_button(
                            label="📥 Download GeoJSON DAS",
                            data=gdf_res.to_json(),
                            file_name="hasil_das.geojson",
                            mime="application/json"
                        )
                    else:
                        st.warning("⚠️ Titik klik di luar alur sungai.")
                except Exception as e:
                    st.error(f"Error delineasi: {e}")
        else:
            # Jika engine belum ada (karena saya tidak punya kode engine kakak)
            st.warning("⚠️ Engine hidrologi belum dimuat. Pastikan class Engine diinisialisasi setelah upload DEM.")
else:
    # Jika belum ada file yang diupload
    st.info("👈 Silakan upload file DEM terlebih dahulu pada panel di atas.")
