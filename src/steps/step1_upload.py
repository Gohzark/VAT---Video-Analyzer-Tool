import streamlit as st
import os
import subprocess
import signal_processing.optical_flow_processer as ofp 
import cv2 as cv

def executer_etape1():
    st.title("Outil d'analyse de mouvement 📊")
    st.subheader("Étape 1 : Sélection de la vidéo (Kaggle Dataset)")
    st.write("Bienvenue ! Veuillez choisir une vidéo parmi celles disponibles dans votre dataset Kaggle pour lancer l'analyse.")

    # On initialise la liste des vidéos dans le session_state pour éviter de ré-interroger Kaggle à chaque clic
    if "liste_videos_kaggle" not in st.session_state:
        with st.spinner("🔍 Récupération de la liste des vidéos sur Kaggle..."):
            try:
                # Interroge l'API pour lister les fichiers du dataset
                result = subprocess.run(
                    ["kaggle", "datasets", "files", "tinodolbeau/opticalflow-videos"],
                    capture_output=True, text=True, check=True
                )
                
                # Le retour de Kaggle contient un tableau textuel. On filtre pour ne garder que les .mp4
                lignes = result.stdout.split("\n")
                fichiers_mp4 = []
                for ligne in lignes:
                    elements = ligne.split()
                    if elements:
                        nom_fichier = elements[0]
                        if nom_fichier.endswith(".mp4"):
                            fichiers_mp4.append(nom_fichier)
                
                st.session_state.liste_videos_kaggle = fichiers_mp4
                
            except Exception as e:
                st.error(f"❌ Impossible de lister les fichiers du dataset Kaggle : {e}")
                st.session_state.liste_videos_kaggle = []

    liste_videos = st.session_state.liste_videos_kaggle

    if not liste_videos:
        st.warning("⚠️ Aucune vidéo au format .mp4 n'a été détectée ou trouvée dans votre dataset Kaggle.")
        return

    # sélecteur de vidéo
    video_selectionnee = st.selectbox(
        "Sélectionnez la vidéo à envoyer à Megaflow :",
        options=liste_videos,
        index=0
    )

    if video_selectionnee:
        st.info(f"🎥 Vidéo sélectionnée pour le traitement : `{video_selectionnee}`")
        st.session_state.video_path = video_selectionnee
        st.session_state.step_over = True
        