import streamlit as st
import os
import time
import signal_processing.optical_flow_processer as ofp
import utils.enums as enums

def executer_etape1():
    st.title("Outil d'analyse de mouvement 📊")
    st.subheader("Étape 1 : Chargement de la vidéo")
    st.write("Bienvenue ! Pour commencer l'analyse de flux optique, veuillez téléverser un fichier vidéo.")

    fichier_charge = st.file_uploader("Choisir une vidéo...", type=["mp4"])

    if fichier_charge is not None:
        barre_progression = st.progress(0, text="Sauvegarde du fichier sur le serveur...")
        
        os.makedirs("tmp", exist_ok=True)
        chemin_video = os.path.join("tmp", fichier_charge.name)
        
        with open(chemin_video, "wb") as f:
            f.write(fichier_charge.read())
                    
        for pourcentage in range(0, 101, 25):
            time.sleep(0.1)
            barre_progression.progress(pourcentage, text=f"Chargement : {pourcentage}%")
        
        barre_progression.empty()
        st.success(f"✅ Fichier '{fichier_charge.name}' correctement chargé !")

        st.write("---")
        if st.button("Passer au traitement ➡️", use_container_width=True):
            st.session_state.video_path = chemin_video
            st.session_state.step = 2
            st.rerun()
            
        with st.expander("👀 Aperçu de la vidéo téléversée", expanded=True):
            st.video(chemin_video)

        