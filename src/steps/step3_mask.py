import streamlit as st
import os

from utils import enums

def executer_etape3():
    
    st.title("⚙️ Configuration du traitement")
    st.subheader("Étape 3 : Choix du masque")
        
    st.info(f"Vidéo active : `{os.path.basename(st.session_state.video_path)}`")
    st.info(f"Algorithme sélectionné : `{st.session_state.algorithm.name}`")
    
    # --- GRILLE DES DESCRIPTIONS ---
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### Aucun masque")
            st.write("**Conserve tous les pixels.**")
            st.caption("Le flux optique sera calculé sur toute l'image, ce qui peut être plus lent et moins précis si la vidéo contient beaucoup de bruit ou de mouvements parasites, mais conserve toute l'information.")
            st.write("") 

    with col2:
        with st.container(border=True):
            st.markdown("### MOG2 (Mixture de Gaussiennes)")
            st.write("**Masque de fond dynamique.**")
            st.caption("Utilise un algorithme de soustraction de fond pour isoler les objets en mouvement. Permet de se concentrer sur les éléments dynamiques de la scène et d'ignorer les zones statiques ou avec du mouvement parasite, ce qui peut améliorer la précision du flux optique.")
        

    # --- ALIGNEMENT DES BOUTONS ---
    col_b1, col_b2 = st.columns(2)

    bouton_gris = (st.session_state.algorithm == enums.Algorithm.Megaflow) 
    
    with col_b1:
        if st.button("❌", key="btn_lk",use_container_width=True):
            st.session_state.mask = enums.Mask.NoMask
            st.session_state.step = 4
            st.rerun()

    with col_b2:
        if st.button("🥽",disabled=bouton_gris, key="btn_fb", use_container_width=True):
            st.session_state.mask = enums.Mask.MOG2
            st.session_state.step = 4
            st.rerun()
            
    if (bouton_gris):
        st.warning("Le masque de mouvement n'est pas compatible avec l'algorithme Megaflow.")
        
    st.write("---")
    if st.button("⬅️ Étape précédente", use_container_width=True):
        st.session_state.step = 2
        st.rerun()