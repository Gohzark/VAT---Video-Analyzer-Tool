from functools import partial
from time import sleep
from signal_processing.optical_flow_processer import getOpticalFlow
import numpy as np
import cv2 as cv
import streamlit as st
import os

def mettre_a_jour_barre(current_frame, total_frames, barre_progression):
    if total_frames <= 0: return
    pourcentage = int((current_frame / total_frames) * 100)
    if current_frame % 5 == 0 or current_frame == total_frames:
        barre_progression.progress(pourcentage, text=f"Images : {current_frame}/{total_frames} ({pourcentage}%)")
                
def mettre_a_jour_image(frame_bgr, cadre_video):
    # Streamlit préfère le RGB, OpenCV utilise le BGR : on convertit
    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    cadre_video.image(frame_rgb, channels="RGB", use_container_width=True)
            
def executer_etape5():
    
    st.info(f"Vidéo active : `{os.path.basename(st.session_state.video_path)}`")
    st.info(f"Algorithme sélectionné : `{st.session_state.algorithm.name}`")
    st.info(f"Masque sélectionné : `{st.session_state.mask.name}`")
    st.info(f"Centrage sélectionné : `{st.session_state.centering.name}`")
    
    st.title("⚙️ Configuration du traitement")
    st.subheader("Étape 5 : Calcul du flux optique")
    
    # 📂 Définition du dossier et chemin de sortie
    dossier_sortie = os.path.join(
        "outputs", 
        os.path.basename(st.session_state.video_path), 
        st.session_state.algorithm.value, 
        st.session_state.mask.value, 
        st.session_state.centering.value
    )
    path_optical_flow = os.path.join(dossier_sortie, "optical_flow.npy")
    cadre_video = st.empty()
    def lancer_le_calcul():
        # Indicateur de démarrage pour vérifier que le bouton déclenche bien l'action
        st.info("Démarrage du calcul du flux optique...")
        barre_progression = st.progress(0, text="Analyse en cours...")
        print("[step5] lancer_le_calcul appelé")
                
        # Vérifier l'existence du fichier temporaire avant d'appeler la fonction
        video_tmp_path = "tmp/" + os.path.basename(st.session_state.video_path)
        if not os.path.exists(video_tmp_path):
            st.error(f"Fichier introuvable : {video_tmp_path} — assurez-vous que la vidéo a bien été téléchargée dans tmp/")
            print(f"[step5] fichier tmp manquant: {video_tmp_path}")
            return

        try:
            optical_flow = getOpticalFlow(
                video_tmp_path,
                st.session_state.algorithm, 
                st.session_state.mask, 
                st.session_state.centering,
                callback_progress=partial(mettre_a_jour_barre, barre_progression=barre_progression),
                callback_image=partial(mettre_a_jour_image, cadre_video=cadre_video)
            )
        except SystemExit:
            st.error("Erreur lors de l'ouverture de la vidéo (openVideo a quitté le processus). Vérifiez le fichier et les codecs.")
            print("[step5] getOpticalFlow a appelé sys.exit lors de l'ouverture de la vidéo")
            return
        except Exception as e:
            st.error(f"Erreur durant le calcul du flux optique : {e}")
            print(f"[step5] Exception getOpticalFlow: {e}")
            return
     
        if optical_flow is not None:
            sleep(2)
            os.makedirs(dossier_sortie, exist_ok=True)
            np.save(path_optical_flow, optical_flow)
            barre_progression.empty()
            st.success(f"Calcul terminé et enregistré dans : {path_optical_flow}")
            st.session_state.calcul_optical_flow_over = True
            st.session_state.step = 6
            st.rerun()
            
        else:
            st.error("Le calcul du flux optique a échoué : résultat None")
            print("[step5] getOpticalFlow a retourné None")

    # 🤖 LOGIQUE D'AFFICHAGE DES BOUTONS
    if os.path.exists(path_optical_flow):
        # Le fichier existe déjà : on affiche un avertissement et on demande confirmation
        st.warning(f"⚠️ Le fichier `{path_optical_flow}` existe déjà.")
        
        if st.button("🔥 Oui, recalculer et écraser", type="primary", use_container_width=True):
            lancer_le_calcul()
        
    else:
        # Le fichier n'existe pas : on affiche le bouton de lancement normal
        if st.button("🚀 Lancer l'analyse du flux optique", use_container_width=True):
            lancer_le_calcul()

    st.write("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Étape précédente", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col_b2:
        if st.session_state.get("calcul_optical_flow_over") or os.path.exists(path_optical_flow):
            if st.button("➡️ Étape suivante", use_container_width=True):
                st.session_state.step = 6
                st.rerun()