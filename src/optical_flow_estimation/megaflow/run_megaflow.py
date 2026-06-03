import os
import subprocess
import json
import shutil
import time

DATASET_SLUG = "tinodolbeau/params-megaflow"

def run_megaflow(chemin_video, centering):
    video_name = os.path.basename(chemin_video)
    DOSSIER_PRINCIPAL = os.path.dirname(os.path.abspath(__file__))

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
    print("✅ Paramètres envoyés ! Prêt pour le calcul sur le navigateur.")