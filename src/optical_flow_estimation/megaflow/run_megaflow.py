import os
import subprocess
import json
import shutil
import time

KERNEL_SLUG = "tinodolbeau/megaflow"
DATASET_SLUG = "tinodolbeau/params-megaflow"

def run_megaflow(chemin_video, centering):
    video_name = os.path.basename(chemin_video)
    DOSSIER_PRINCIPAL = os.path.dirname(os.path.abspath(__file__))

    # 1. Préparation et envoi des paramètres
    DOSSIER_DATASET = os.path.join(DOSSIER_PRINCIPAL, "dataset_params")
    if os.path.exists(DOSSIER_DATASET):
        shutil.rmtree(DOSSIER_DATASET)
    os.makedirs(DOSSIER_DATASET)

    params = {"VIDEO": video_name, "CENTERING": centering.value}
    with open(os.path.join(DOSSIER_DATASET, "params.json"), "w") as f:
        json.dump(params, f)
    with open(os.path.join(DOSSIER_DATASET, "dataset-metadata.json"), "w") as f:
        json.dump({"id": DATASET_SLUG, "title": "params-megaflow"}, f)

    print("🚀 Mise à jour des paramètres sur Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "version", "-p", DOSSIER_DATASET, "-m", "Maj params", "--dir-mode", "tar"],
        check=True,
        stdout=subprocess.DEVNULL
    )

    # 2. Phase manuelle sur le navigateur
    print("👉 Va sur https://www.kaggle.com/code/tinodolbeau/megaflow")
    print("  - Clique sur 'Run All' pour lancer le calcul du flux optique.")
    input("👉 UNE FOIS LE CALCUL FINI (optical_flow.npy généré), appuie sur Entrée pour télécharger...")

    # 3. Téléchargement direct sans poser de questions
    dossier_cible = os.path.join("outputs", video_name, "Megaflow", "NoMask", centering.value)
    os.makedirs(dossier_cible, exist_ok=True)

    print("📥 Téléchargement du flux optique...")
    subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", dossier_cible],
        check=True,
        stdout=subprocess.DEVNULL
    )

    chemin_attendu = os.path.join(dossier_cible, "optical_flow.npy")
    if not os.path.exists(chemin_attendu):
        contenu = os.listdir(dossier_cible) if os.path.exists(dossier_cible) else []
        raise FileNotFoundError(f"optical_flow.npy absent du dossier local. Contenu reçu : {contenu}")
    
    # 🧹 NETTOYAGE DES FICHIERS INUTILES
    print("🧹 Nettoyage des fichiers secondaires téléchargés...")
    for item in os.listdir(dossier_cible):
        chemin_item = os.path.join(dossier_cible, item)
        # On garde uniquement notre fichier .npy, on vire tout le reste (dossiers ou fichiers)
        if item != "optical_flow.npy":
            if os.path.isdir(chemin_item):
                shutil.rmtree(chemin_item)
            else:
                os.remove(chemin_item)

    print(f"🎉 Flux optique récupéré avec succès dans {dossier_cible} !")