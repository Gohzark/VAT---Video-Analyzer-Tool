from signal_processing.optical_flow_processer import flowToMagnitudes
import numpy as np
import streamlit as st
import os
import optical_flow_estimation.optical_flow_Farneback as optical_flow_Farneback
import optical_flow_estimation.optical_flow_LK as optical_flow_LK   
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
            st.caption("Spécifique au mouvements entrecoupés de pauses.")
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
            st.session_state.step = 8
            st.rerun()

    with col_b2:
        if st.button("📡", key="btn_fb", use_container_width=True):
            st.session_state.analysis = enums.Analyze.FFT
            st.session_state.step = 8
            st.rerun()

    with col_b3:
        if st.button("🛝", key="btn_mf", use_container_width=True):
            st.session_state.analysis = enums.Analyze.Sliding
            st.session_state.step = 8
            st.rerun()
        
    # TODO: continuer
    if (st.session_state.analysis is not None):
        analyse = initAnalyse(st.session_state.analysis, st.session_state.video_path, cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH), st.session_state.algorithm, st.session_state.mask, st.session_state.centering)
        analyse.detectMovements(cap.get(cv.CAP_PROP_FPS))
        
    st.write("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Étape précédente", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    with col_b2:
        if st.session_state.get("analysis_over") or os.path.exists():
            if st.button("➡️ Étape suivante", use_container_width=True):
                st.session_state.step = 7
                st.rerun()