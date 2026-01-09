# ... (Bagian atas kode HydroEngine tetap sama, tidak perlu diubah) ...

    # TAMPILKAN PETA
    if 'active_dem' in st.session_state:
        st.subheader("2. Peta Interaktif")
        
        # UPDATE: Set default ke Satellite Hybrid agar sungai terlihat jelas
        m = leafmap.Map(google_map="HYBRID") 
        
        # Coba visualisasi DEM (Jika ringan akan muncul, jika berat di-skip)
        try:
            m.add_raster(st.session_state['active_dem'], layer_name="Topografi (DEM)", colormap="terrain", opacity=0.6)
        except:
            # Jika gagal visualisasi, beri info tapi JANGAN STOP program
            st.caption("ℹ️ Visualisasi warna topografi dinonaktifkan demi performa. Silakan klik berdasarkan panduan Peta Satelit.")

        # Tampilkan Batas Wilayah (Jika ada)
        if up_boundary:
            try:
                fiona.drvsupport.supported_drivers['KML'] = 'rw'
                fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{up_boundary.name.split('.')[-1]}") as tmp:
                    tmp.write(up_boundary.getbuffer())
                    tmp_path = tmp.name
                
                gdf = gpd.read_file(tmp_path)
                bounds = gdf.total_bounds
                m.zoom_to_bounds((bounds[0], bounds[1], bounds[2], bounds[3]))
                
                # Tampilkan garis batas dengan warna cerah
                style = {'fillColor': '#00000000', 'color': 'cyan', 'weight': 3}
                m.add_gdf(gdf, layer_name="Batas Wilayah", style=style)
            except: pass
            
        map_out = st_folium(m, height=500, width=None)
        
        # LOGIKA KLIK (Tetap Jalan Normal)
        if map_out and map_out['last_clicked']:
            lat = map_out['last_clicked']['lat']
            lng = map_out['last_clicked']['lng']
            st.info(f"📍 Koordinat Klik: {lat:.5f}, {lng:.5f}")
            
            if 'engine' in st.session_state:
                with st.spinner("⏳ Sedang menghitung batas DAS..."):
                    eng = st.session_state['engine']
                    poly_das = eng.delineate(lng, lat)
                    
                    if poly_das:
                        area_km2 = (poly_das.area * 111.32 * 111.32) # Estimasi kasar area
                        st.success(f"✅ DAS Berhasil Dibuat! Luas: ±{area_km2:.2f} km²")
                        
                        # Siapkan Download
                        gdf_res = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly_das])
                        st.download_button(
                            label="📥 Download GeoJSON DAS",
                            data=gdf_res.to_json(),
                            file_name="hasil_das.geojson",
                            mime="application/json"
                        )
                    else:
                        st.warning("⚠️ Titik klik di luar alur sungai. Coba klik pas di tengah lembah/sungai pada peta satelit.")
