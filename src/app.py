import shutil
import sys
import os
import utils.enums as enums
import streamlit as st
from steps.step1_upload import executer_etape1
from steps.step2_algo import executer_etape2
from steps.step3_mask import executer_etape3
from steps.step4_centering import executer_etape4

# Gestion des chemins pour éviter les erreurs d'import de modules locaux
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Outil d'analyse de mouvement 📊", layout="centered")

# --- INITIALISATION DE LA SESSION STATE ---
if "step" not in st.session_state:
    st.session_state.step = 1

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "algorithm" not in st.session_state:
    st.session_state.algorithm = enums.Algorithm.Farneback
    
if "mask" not in st.session_state:
    st.session_state.mask = enums.Mask.NoMask
    
if "centering" not in st.session_state:
    st.session_state.centering = enums.Centering.NoCentering

# --- ROUTEUR D'ÉTAPES ---
if st.session_state.step == 1:
    executer_etape1()

elif st.session_state.step == 2:
    executer_etape2()

elif st.session_state.step == 3:
    executer_etape3()

elif st.session_state.step == 4:
    executer_etape4()

# --- BOUTON DE RETOUR GLOBAL ---
if st.session_state.step > 1:
    st.write("---")
    if st.button("⬅️ Changer de vidéo", use_container_width=True):
        
        # NETTOYAGE DU DOSSIER TMP
        if os.path.exists("tmp"):
            try:
                shutil.rmtree("tmp")  # Supprime le dossier 'tmp' et tout ce qu'il contient
            except Exception as e:
                # Évite que l'application ne crashe si le fichier est en cours d'utilisation par OpenCV
                st.sidebar.error(f"Erreur lors de la suppression de tmp: {e}")

        # Réinitialisation des variables de la session
        st.session_state.step = 1
        st.session_state.video_path = None
        st.session_state.algorithm = enums.Algorithm.Farneback # Remis à la valeur par défaut
        st.session_state.mask = enums.Mask.NoMask
        st.session_state.centering = enums.Centering.NoCentering
        st.rerun()
        
