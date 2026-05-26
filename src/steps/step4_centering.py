import streamlit as st
import os

from utils import enums

def executer_etape4():
    
    st.info(f"Vidéo active : `{os.path.basename(st.session_state.video_path)}`")
    st.info(f"Algorithme sélectionné : `{st.session_state.algorithm.name}`")
    st.info(f"Masque sélectionné : `{st.session_state.mask.name}`")
    
    st.title("⚙️ Configuration du traitement")
    st.subheader("Étape 4 : Centrage du flux optique")
        
    # --- GRILLE DES DESCRIPTIONS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### Aucun centrage")
            st.write("**Laisse le cadre fixe pour analyser le flux optique sur l'ensemble de l'image.**")
            st.caption("Pour étudier le déplacement global du sujet dans l'espace.")
            st.write("") 

    with col2:
        with st.container(border=True):
            st.markdown("### EMA (Exponential Moving Average)")
            st.write("**Amortit le recentrage en calculant une moyenne fluide basée sur les positions récentes de l'objet.**")
            st.caption("Pour suivre des mouvements fluides.")
        
    with col3:
        with st.container(border=True):
            st.markdown("### Kalman")
            st.write("**Prédit et estime la trajectoire de l'objet pour corriger le bruit et les pertes de détection du masque.**")
            st.caption("Pour les mouvements rapides et pour maintenir un centrage stable même si le sujet est temporairement masqué.")
                       
    # --- ALIGNEMENT DES BOUTONS ---
    col_b1, col_b2, col_b3 = st.columns(3)

    bouton_gris = (st.session_state.algorithm == enums.Algorithm.Megaflow) 
    
    with col_b1:
        if st.button("🖼️", key="btn_lk",use_container_width=True):
            st.session_state.centering = enums.Centering.NoCentering
            st.session_state.step = 5
            st.rerun()

    with col_b2:
        if st.button("🦤",disabled=bouton_gris, key="btn_fb", use_container_width=True):
            st.session_state.centering = enums.Centering.ExponentialMovingAverage
            st.session_state.step = 5
            st.rerun()
            
    with col_b3:
        if st.button("🦅",disabled=bouton_gris, key="btn_kalman", use_container_width=True):
            st.session_state.centering = enums.Centering.Kalman
            st.session_state.step = 5
            st.rerun()
            
    if (bouton_gris):
        st.warning("Le centrage n'est pas compatible avec l'algorithme Megaflow.")
        
    st.write("---")
    if st.button("⬅️ Étape précédente", use_container_width=True):
        st.session_state.step = 3
        st.rerun()