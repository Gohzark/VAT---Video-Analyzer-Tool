import shutil
import sys
import os
from steps.step7_analysis import executer_etape7
import utils.enums as enums
import streamlit as st
from steps.step1_upload import executer_etape1
from steps.step2_algo import executer_etape2
from steps.step3_mask import executer_etape3
from steps.step4_centering import executer_etape4
from steps.step5_optical_flow import executer_etape5
from steps.step6_magn import executer_etape6

# Gestion des chemins pour éviter les erreurs d'import de modules locaux
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Outil d'analyse de mouvement 📊", layout="wide")

# --- INITIALISATION DE LA SESSION STATE ---
if "step" not in st.session_state:
    st.session_state.step = 1
    
if "step_over" not in st.session_state:
    st.session_state.step_over = False

if "video_path" not in st.session_state:
    st.session_state.video_path = None


if "fps" not in st.session_state:
    st.session_state.fps = None

if "algorithm" not in st.session_state:
    st.session_state.algorithm = enums.Algorithm.Farneback
    
if "mask" not in st.session_state:
    st.session_state.mask = enums.Mask.NoMask
    
if "centering" not in st.session_state:
    st.session_state.centering = enums.Centering.NoCentering
    
if "analysis" not in st.session_state:
    st.session_state.analysis = None
    
# --- SÉCURITÉ FPS GLOBALE ---
# Si une vidéo est chargée mais que les FPS ont sauté au rechargement (fichiers déjà existants)
if st.session_state.video_path and (st.session_state.fps is None or st.session_state.fps == 0):
    video_tmp_path = "tmp/" + os.path.basename(st.session_state.video_path)
    if os.path.exists(video_tmp_path):
        import cv2 as cv
        video = cv.VideoCapture(video_tmp_path)
        st.session_state.fps = video.get(cv.CAP_PROP_FPS)
        video.release()
    
# --- ROUTEUR D'ÉTAPES ---
if st.session_state.step == 1:
    executer_etape1()

elif st.session_state.step == 2:
    executer_etape2()

elif st.session_state.step == 3:
    executer_etape3()

elif st.session_state.step == 4:
    executer_etape4()
    
elif st.session_state.step == 5:
    executer_etape5()

elif st.session_state.step == 6:
    executer_etape6()
    
elif st.session_state.step == 7:
    executer_etape7()
    
st.write("---")
col_b1, col_b2 = st.columns(2)
with col_b1:
    if st.button("⬅️ Étape précédente", use_container_width=True):
        if st.session_state.step == 5 and st.session_state.algorithm == enums.Algorithm.Megaflow:
            st.session_state.step = 2
        else:
            st.session_state.step -= 1  
        st.rerun()
        
with col_b2:
    if (st.session_state.step_over):
        if st.button("➡️ Étape suivante", use_container_width=True):
            st.session_state.step_over = False
            if st.session_state.step == 2 and st.session_state.algorithm == enums.Algorithm.Megaflow:
                st.session_state.step = 5
            else:
                st.session_state.step += 1
            st.rerun()
                
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
        st.session_state.step_over = False
        st.session_state.video_path = None
        st.session_state.fps = None
        st.session_state.algorithm = enums.Algorithm.Farneback # Remis à la valeur par défaut
        st.session_state.mask = enums.Mask.NoMask
        st.session_state.centering = enums.Centering.NoCentering
        st.rerun()
        
