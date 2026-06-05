import subprocess
import streamlit as st
import os
import utils.enums as enums

def executer_etape2():
    
    def download_video(nom_fichier):
        nom_brut = os.path.basename(nom_fichier)
        chemin_local = os.path.join("tmp", nom_brut)
        # Si la vidéo n'est pas déjà dans le dossier tmp, on la télécharge depuis Kaggle
        if not os.path.exists(chemin_local):
            os.makedirs("tmp", exist_ok=True)
            with st.spinner(f"📥 Téléchargement de `{nom_fichier}` depuis Kaggle pour le traitement local..."):
                try:
                    # On cible uniquement le fichier précis pour éviter de télécharger tout le dataset
                    subprocess.run([
                        "kaggle", "datasets", "download", 
                        "tinodolbeau/opticalflow-videos", "-f", nom_brut, "-p", "tmp"
                    ], check=True)
                    
                    if os.path.exists(chemin_local):
                        st.success(f"✅ `{nom_brut}` récupéré avec succès !")
                    else:
                        st.error(f"❌ Le fichier `{nom_brut}` n'est pas apparu dans le dossier tmp.")
                        return None
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du téléchargement de la vidéo : {e}")
                    return None
        return chemin_local

    st.title("⚙️ Configuration du traitement")
    st.subheader("Étape 2 : Choix de l'algorithme de flux optique")
    st.info(f"Vidéo active : `{os.path.basename(st.session_state.video_path)}`")
        
    # --- GRILLE DES DESCRIPTIONS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 🎯 Lucas-Kanade")
            st.write("**Flux optique Sparse.**")
            st.caption("Peu coûteux en ressources, mais ne suit que certains points clés et peut rater l'information.")
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
    algo_selectionne = None

    with col_b1:
        if st.button("🎯", key="btn_lk", width='stretch'):
            algo_selectionne = enums.Algorithm.LucasKanade

    with col_b2:
        if st.button("🕸️", key="btn_fb", width='stretch'):
            algo_selectionne = enums.Algorithm.Farneback

    with col_b3:
        if st.button("⚡", key="btn_mf", width='stretch'):
            algo_selectionne = enums.Algorithm.Megaflow

    if algo_selectionne is not None:
        chemin_pret = download_video(st.session_state.video_path)
        
        if chemin_pret:
            st.session_state.video_path = chemin_pret
            st.session_state.algorithm = algo_selectionne
            st.session_state.step_over = True
            st.rerun()

    if st.session_state.step_over:
        st.info(f"🎥 Algorithme de flux optique sélectionné : `{st.session_state.algorithm.name}`")