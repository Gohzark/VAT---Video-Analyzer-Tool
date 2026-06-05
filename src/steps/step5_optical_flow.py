from functools import partial
import shutil
import subprocess
from signal_processing.optical_flow_processer import getOpticalFlow
import numpy as np
import cv2 as cv
import streamlit as st
import os


def mettre_a_jour_barre(current_frame, total_frames, barre_progression):
    if total_frames <= 0:
        return
    pourcentage = int((current_frame / total_frames) * 100)
    if current_frame % 5 == 0 or current_frame == total_frames:
        barre_progression.progress(
            pourcentage,
            text=f"Images : {current_frame}/{total_frames} ({pourcentage}%)",
        )


def mettre_a_jour_image(frame_bgr, cadre_video):
    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    cadre_video.image(frame_rgb, channels="RGB", width='stretch')


def executer_etape5():

    st.title("Étape 5 — Calcul du flux optique")

    st.info(f"Vidéo active : `{os.path.basename(st.session_state.video_path)}`")
    st.info(f"Algorithme sélectionné : `{st.session_state.algorithm.name}`")
    st.info(f"Masque sélectionné : `{st.session_state.mask.name}`")
    st.info(f"Centrage sélectionné : `{st.session_state.centering.name}`")


    st.divider()

    dossier_sortie = os.path.join(
        "outputs",
        os.path.basename(st.session_state.video_path),
        st.session_state.algorithm.value,
        st.session_state.mask.value,
        st.session_state.centering.value,
    )
    path_optical_flow = os.path.join(dossier_sortie, "optical_flow.npy")

    cadre_video = st.empty()

    def lancer_le_calcul():
        barre_progression = st.progress(0, text="Démarrage du calcul…")

        video_tmp_path = "tmp/" + os.path.basename(st.session_state.video_path)
        if not os.path.exists(video_tmp_path):
            st.error(
                f"Fichier vidéo introuvable : `{video_tmp_path}`  \n"
                "Assurez-vous que la vidéo a bien été téléversée dans `tmp/`."
            )
            return

        try:
            res_flow, res_fps = getOpticalFlow(
                video_tmp_path,
                st.session_state.algorithm,
                st.session_state.mask,
                st.session_state.centering,
                callback_progress=partial(
                    mettre_a_jour_barre, barre_progression=barre_progression
                ),
                callback_image=partial(mettre_a_jour_image, cadre_video=cadre_video),
            )
        except SystemExit:
            st.error("Impossible d'ouvrir la vidéo.")
            return
        except Exception as e:
            st.error(f"Erreur durant le calcul du flux optique : {e}")
            return

        if st.session_state.algorithm.name == "Megaflow":
            if res_fps is not None:
                st.session_state.fps = res_fps
                barre_progression.empty()
                # Enregistre le message pour le rechargement
                st.session_state.megaflow_success_msg = (
                    "Paramètres envoyés à Kaggle. "
                    "Lancez le notebook puis revenez télécharger le résultat."
                )
                st.rerun()
            else:
                st.error("Impossible de récupérer les FPS pour initialiser Megaflow.")
        else:
            if res_flow is not None and res_fps is not None:
                os.makedirs(dossier_sortie, exist_ok=True)
                np.save(path_optical_flow, res_flow)
                st.session_state.fps = res_fps
                st.session_state.step_over = True
                st.session_state.pop("forcing_recalcul", None)
                barre_progression.empty()
                st.success(f"Calcul terminé — résultat enregistré dans `{path_optical_flow}`")
                st.rerun()
            else:
                manquants = []
                if res_flow is None:
                    manquants.append("flux optique (`res_flow`)")
                if res_fps is None:
                    manquants.append("FPS (`res_fps`)")
                st.error(f"Le calcul a échoué — valeur(s) manquante(s) : {', '.join(manquants)}")

    # ── Branche A : fichier existant, pas de recalcul forcé ──────────────────
    if os.path.exists(path_optical_flow) and "forcing_recalcul" not in st.session_state:

        st.success("Flux optique disponible en local")
        st.caption(f"`{path_optical_flow}`")

        st.write("Voulez-vous relancer le calcul ?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Conserver et continuer →", width='stretch', type="primary"):
                st.session_state.step_over = True
                st.rerun()

        with col2:
            if st.button("Recalculer (sauvegarde .bak)", width='stretch'):
                st.session_state.forcing_recalcul = True
                st.session_state.step_over = False
                backup_path = path_optical_flow + ".bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(path_optical_flow, backup_path)
                st.sidebar.info("Ancien flux sauvegardé en `.npy.bak`.")
                st.rerun()

    # ── Branche B : calcul à effectuer (ou recalcul forcé) ───────────────────
    else:

        # ── Megaflow (distant) ────────────────────────────────────────────────
        if st.session_state.algorithm.name == "Megaflow":

            st.info(
                "Le calcul Megaflow s'effectue sur Kaggle (≈ 10–15 s / image).  \n"
                "Cliquez sur **Envoyer les paramètres**, lancez le notebook, "
                "puis revenez télécharger le résultat."
            )

            if "megaflow_success_msg" in st.session_state:
                st.success(st.session_state.megaflow_success_msg)
        
            if st.button("Envoyer les paramètres à Kaggle", type="primary", width='stretch'):
                with st.spinner("Communication avec Kaggle en cours..."):
                    lancer_le_calcul()

            st.link_button(
                "Ouvrir le notebook Kaggle Megaflow",
                "https://www.kaggle.com/code/tinodolbeau/megaflow",
                width='stretch',
            )

            st.divider()

            if st.button("Télécharger le flux depuis Kaggle", width='stretch'):
                try:
                    os.makedirs(dossier_sortie, exist_ok=True)
                    commande = f'kaggle kernels output tinodolbeau/megaflow -p "{dossier_sortie}"'
                    resultat = subprocess.run(commande, shell=True, capture_output=True, text=True)

                    if resultat.returncode != 0:
                        st.error("Le téléchargement automatique a échoué.")
                        with st.expander("Récupérer le fichier manuellement", expanded=True):
                            st.markdown(f"""
                                1. Dans l'onglet Kaggle, déroulez **Output** → téléchargez `optical_flow.npy`.
                                2. Dans votre terminal, déplacez le fichier :
                                ```bash
                                mv ~/Téléchargements/optical_flow.npy "{os.path.abspath(path_optical_flow)}"
                                ```
                                3. Appuyez sur **R** pour rafraîchir l'application.
                            """)
                    else:
                        # Nettoyage des fichiers parasites
                        for item in os.listdir(dossier_sortie):
                            chemin_item = os.path.join(dossier_sortie, item)
                            if item != "optical_flow.npy":
                                if os.path.isdir(chemin_item):
                                    shutil.rmtree(chemin_item)
                                else:
                                    os.remove(chemin_item)

                        st.session_state.step_over = True
                        st.session_state.pop("forcing_recalcul", None)
                        st.success("Flux optique récupéré depuis Kaggle.")
                        st.rerun()

                except Exception as e:
                    st.error(f"Erreur inattendue : {e}")

        # ── Farneback / Lucas-Kanade (local) ─────────────────────────────────
        else:
            st.write(
                f"L'algorithme **{st.session_state.algorithm.name}** "
                "va s'exécuter localement sur votre machine."
            )

            if st.button(
                f"Lancer le calcul — {st.session_state.algorithm.name}",
                type="primary",
                width='stretch',
            ):
                lancer_le_calcul()