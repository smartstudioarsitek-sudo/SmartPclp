# --- TAB 2: LONG SECTION ---
with tabs[1]:
    st.subheader("Long Section (Profil Memanjang)")
    st.info("Support: Excel (.xls/.xlsx) dan CSV (.csv)")
    
    # Update tipe file agar bisa terima CSV juga
    f_long = st.file_uploader("Upload Data Long Section", type=['xls', 'xlsx', 'csv'], key='long_up')
    
    if f_long:
        pts_ogl, pts_dsn = [], []
        
        # Deteksi Jenis File & Baca Data
        try:
            # Skenario 1: File adalah CSV
            if f_long.name.lower().endswith('.csv'):
                df = pd.read_csv(f_long)
                # Asumsi CSV Long Section formatnya: Jarak, Elevasi (2 kolom pertama)
                df = df.select_dtypes(include=[np.number]).dropna()
                if df.shape[1] >= 2:
                    pts_ogl = df.iloc[:, :2].values.tolist()
                    pts_ogl.sort(key=lambda x: x[0])
                    st.success("File CSV berhasil dibaca sebagai Tanah Asli!")
                else:
                    st.warning("CSV terbaca, tapi tidak ditemukan 2 kolom angka (Jarak, Elevasi).")

            # Skenario 2: File adalah Excel (.xls/.xlsx)
            else:
                try:
                    xls_l = pd.ExcelFile(f_long)
                    sheets = xls_l.sheet_names
                    
                    c_sel1, c_sel2 = st.columns(2)
                    s_long_ogl = c_sel1.selectbox("Sheet Tanah (Long)", ["[Pilih]"] + sheets)
                    s_long_dsn = c_sel2.selectbox("Sheet Desain (Long)", ["[Pilih]"] + sheets)
                    
                    if st.button("RUN LONG SECTION"):
                        if s_long_ogl != "[Pilih]":
                            df = pd.read_excel(f_long, sheet_name=s_long_ogl)
                            df = df.select_dtypes(include=[np.number]).dropna()
                            if df.shape[1] >= 2:
                                pts_ogl = df.iloc[:, :2].values.tolist()
                                pts_ogl.sort(key=lambda x: x[0])
                        
                        if s_long_dsn != "[Pilih]":
                            df = pd.read_excel(f_long, sheet_name=s_long_dsn)
                            df = df.select_dtypes(include=[np.number]).dropna()
                            if df.shape[1] >= 2:
                                pts_dsn = df.iloc[:, :2].values.tolist()
                                pts_dsn.sort(key=lambda x: x[0])

                except ValueError:
                    # Fallback: Jika error ValueError, kemungkinan ini file CSV/Text yang diberi nama .xls
                    st.warning("⚠️ File ini sepertinya bukan Excel murni, mencoba membaca sebagai Text/CSV...")
                    f_long.seek(0) # Reset pointer
                    df = pd.read_csv(f_long, sep=None, engine='python') # Auto-detect separator
                    df = df.select_dtypes(include=[np.number]).dropna()
                    if df.shape[1] >= 2:
                        pts_ogl = df.iloc[:, :2].values.tolist()
                        pts_ogl.sort(key=lambda x: x[0])
                        st.success("Berhasil dibaca dengan metode Fallback (Text Mode)!")

            # Simpan hasil ke session state
            if pts_ogl or pts_dsn:
                st.session_state['long_res'] = (pts_ogl, pts_dsn)

        except Exception as e:
            st.error(f"Gagal membaca file: {str(e)}")

    # Bagian Visualisasi (Plotting)
    if 'long_res' in st.session_state:
        ogl, dsn = st.session_state['long_res']
        
        if not ogl and not dsn:
            st.warning("Data kosong. Pastikan file Excel/CSV memiliki kolom angka.")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Plot Tanah
            if ogl: 
                ax.plot(*zip(*ogl), 'k--', label='Tanah Asli')
                ax.fill_between([p[0] for p in ogl], [p[1] for p in ogl], min([p[1] for p in ogl])-5, color='gray', alpha=0.1)
                
            # Plot Desain
            if dsn: 
                ax.plot(*zip(*dsn), 'r-', label='Desain', linewidth=2)
            
            ax.set_title("Profil Memanjang (Long Section)")
            ax.set_xlabel("Jarak Kumulatif (m)")
            ax.set_ylabel("Elevasi (m)")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            st.pyplot(fig)
            
            # Download DXF Long
            dxf_long = generate_dxf((ogl, dsn), mode="long")
            st.download_button("📥 Download DXF Long Section", dxf_long, "Long_Section.dxf", "application/dxf")
