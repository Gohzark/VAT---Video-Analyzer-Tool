import streamlit as st
import os
import utils.enums as enums

def executer_etape2():
    
    st.title("⚙️ Configuration du traitement")
    st.subheader("Étape 2 : Choix de l'algorithme de flux optique")
    st.info(f"Vidéo active : `{os.path.basename(st.session_state.video_path)}`")
        
    # --- GRILLE DES DESCRIPTIONS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 🎯 Lucas-Kanade")
            st.write("**Flux optique Sparse.**")
            st.caption("Peu coûteux en ressources, mais ne suit que certains points clés et peu rater l'information.")
            st.write("") 

    with col2:
        with st.container(border=True):
            st.markdown("### 🕸️ Farneback")
            st.write("**Flux optique Dense.**")
            st.caption("Calcule le mouvement pour chaque pixel. Offre une bonne qualité d'analyse mais nécessite plus de ressources.")
            
    with col3:
        with st.container(border=True):
            st.markdown("### ⚡ MEGAFLOW")
            st.write("**Deep Learning (Transformers).**")
            st.caption("Modèle préentraîné ultra-performant. Nécessite des vidéos à faible FPS, et très gourmand en ressources.")

    # --- ALIGNEMENT DES BOUTONS ---
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        if st.button("🎯", key="btn_lk",use_container_width=True):
            st.session_state.algorithm = enums.Algorithm.LucasKanade
            st.session_state.step = 3
            st.rerun()

    with col_b2:
        if st.button("🕸️", key="btn_fb", use_container_width=True):
            st.session_state.algorithm = enums.Algorithm.Farneback
            st.session_state.step = 3
            st.rerun()

    with col_b3:
        if st.button("⚡", key="btn_mf", use_container_width=True):
            st.session_state.algorithm = enums.Algorithm.Megaflow
            st.session_state.step = 3
            st.rerun()