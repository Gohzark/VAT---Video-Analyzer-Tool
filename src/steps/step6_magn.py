from signal_processing.optical_flow_processer import flowToMagnitudes
import numpy as np
import streamlit as st
import os
import optical_flow_estimation.optical_flow_Farneback as optical_flow_Farneback
import optical_flow_estimation.optical_flow_LK as optical_flow_LK   
from utils import enums

def executer_etape6():
    st.info(f"Vidéo active : `{os.path.basename(st.session_state.video_path)}`")
    st.info(f"Algorithme sélectionné : `{st.session_state.algorithm.name}`")
    st.info(f"Masque sélectionné : `{st.session_state.mask.name}`")
    st.info(f"Centrage sélectionné : `{st.session_state.centering.name}`")

    st.title("⚙️ Configuration du traitement")
    st.subheader("Étape 6 : Calcul des magnitudes à partir du flux optique")

    dossier_sortie = os.path.join(
        "outputs",
        os.path.basename(st.session_state.video_path),
        st.session_state.algorithm.value,
        st.session_state.mask.value,
        st.session_state.centering.value
    )
    path_flow = os.path.join(dossier_sortie, "optical_flow.npy")
    path_magnitudes = os.path.join(dossier_sortie, "magnitudes.npy")

    if not os.path.exists(dossier_sortie) or not os.path.exists(path_flow):
        st.error("Le fichier de flux optique est introuvable. Assurez-vous d'avoir exécuté l'étape précédente.")
        st.stop()

    def lancer_le_calcul():
        magnitudes = flowToMagnitudes(path_flow)
        if magnitudes is None:
            st.error("Le calcul des magnitudes a échoué.")
            return
        np.save(path_magnitudes, magnitudes)
        st.success(f"Calcul terminé et enregistré dans : {path_magnitudes}")
        st.session_state.calcul_magn_over = True

    # 🤖 LOGIQUE D'AFFICHAGE DES BOUTONS
    if os.path.exists(path_magnitudes):
        st.warning(f"⚠️ Le fichier `{path_magnitudes}` existe déjà.")

        if st.button("🔥 Oui, recalculer et écraser", type="primary", use_container_width=True):
                lancer_le_calcul()
        
    else:
        if st.button("🚀 Lancer le calcul des magnitudes", use_container_width=True):
            lancer_le_calcul()

    st.write("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Étape précédente", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    with col_b2:
        if st.session_state.get("calcul_magn_over") or os.path.exists(path_magnitudes):
            if st.button("➡️ Étape suivante", use_container_width=True):
                st.session_state.step = 7
                st.rerun()