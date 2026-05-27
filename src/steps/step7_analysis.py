from signal_processing.optical_flow_processer import initAnalyse, detectMovements
import streamlit as st
import os 
from utils import enums

def executer_etape7():
    
    st.title("⚙️ Configuration du traitement")
    st.subheader("Étape 7 : Analyse des magnitudes")
    st.info(f"Vidéo active : `{os.path.basename(st.session_state.video_path)}`")
        
    # --- GRILLE DES DESCRIPTIONS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 🏁 Start-Stop")
            st.write("**Détecte les arrêts et les reprises de mouvements.**")
            st.caption("Spécifique aux mouvements entrecoupés de pauses.")
            st.write("") 

    with col2:
        with st.container(border=True):
            st.markdown("### 📡 Fast Fourier Transform (FFT)")
            st.write("**Analyse des fréquences de mouvement.**")
            st.caption("Nécessite une signal stationnaire, donc avec périodicité constante.")
            
    with col3:
        with st.container(border=True):
            st.markdown("### 🛝 Sliding")
            st.write("**Analyse du signal par décalage du signal.**")
            st.caption("Utile pour détecter toute sorte de motifs répétitifs, ne nécessite pas de périodicité uniforme.")

    # --- ALIGNEMENT DES BOUTONS ---
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        if st.button("🏁", key="btn_lk",use_container_width=True):
            st.session_state.analysis = enums.Analyze.StartStop

    with col_b2:
        if st.button("📡", key="btn_fb", use_container_width=True):
            st.session_state.analysis = enums.Analyze.FastFourierTransformation

    with col_b3:
        if st.button("🛝", key="btn_mf", use_container_width=True):
            st.session_state.analysis = enums.Analyze.Sliding
        
        
    st.write("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Étape précédente", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    with col_b2:
        if st.session_state.get("analysis_over") :
            if st.button("➡️ Terminer et revenir au début", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
                
    if (st.session_state.analysis is not None):
        analyse = initAnalyse(os.path.basename(st.session_state.video_path), st.session_state.algorithm, st.session_state.mask, st.session_state.centering, st.session_state.analysis)
        detectMovements(analyse, st.session_state.fps)
        st.session_state.analysis_over = True
        st.success("✅ Analyse " + st.session_state.analysis.value + " terminée ! Résultats disponibles.")
        st.session_state.analysis = None
