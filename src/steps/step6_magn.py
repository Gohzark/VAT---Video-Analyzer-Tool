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
    
    if st.button("🚀 Lancer le calcul des magnitudes", use_container_width=True):
        dossier_sortie = os.path.join("outputs", os.path.basename(st.session_state.video_path), st.session_state.algorithm.value, st.session_state.mask.value, st.session_state.centering.value)
        if not os.path.exists(dossier_sortie):
            raise ValueError(f"Le dossier de sortie n'existe pas : {dossier_sortie}. Assurez-vous d'avoir exécuté l'étape précédente pour générer le flux optique.")
        else:
            path_flow = os.path.join(dossier_sortie, "optical_flow.npy")
            path_magnitudes = os.path.join(dossier_sortie, "magnitudes.npy")
            magnitudes = flowToMagnitudes(path_flow)
            np.save(path_magnitudes, magnitudes)
            st.success("Calcul des magnitudes terminé ! Enregistré dans : " + path_magnitudes)