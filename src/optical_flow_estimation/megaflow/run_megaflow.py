import os
import subprocess
import time
from unittest import result
import nbformat, json

KERNEL_SLUG = "tinodolbeau/megaflow"
DOSSIER = "./"

def run_megaflow(chemin_video, centering, callback_progress, callback_image):
    #Configure le notebook
    params = {"VIDEO": os.path.basename(chemin_video), "CENTERING": centering.value}

    DOSSIER = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(DOSSIER, "megaflow.ipynb"), "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Génère une cellule de params en première position
    nb.cells = [c for c in nb.cells if not c.get("metadata", {}).get("is_params")]
    param_cell = nbformat.v4.new_code_cell(
        "\n".join([f"{k} = {json.dumps(v)}" for k, v in params.items()])
    )
    param_cell["metadata"]["is_params"] = True  # marque la cellule
    nb.cells.insert(0, param_cell)

    with open("megaflow.ipynb", "w") as f:
        nbformat.write(nb, f)
        
    # 1. Pousse et lance le notebook
    print("🚀 Envoi du notebook...")
    subprocess.run(["kaggle", "kernels", "push", "-p", DOSSIER], check=True)

    # 2. Attends la fin
    print("⏳ En cours d'exécution...")
    while True:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", DOSSIER],
            capture_output=True, text=True
        )
        print(result.stdout)
        print(result.stderr)
        status = result.stdout
        print(status.strip())

        if "complete" in status:
            print("✅ Terminé !")
            break
        elif "error" in status:
            print("❌ Erreur lors de l'exécution")
            break

        time.sleep(30)

    # 3. Récupère les outputs
    print("📥 Récupération des outputs...")
    os.makedirs("../../../outputs/megaflow/", exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", "../../../outputs/megaflow/"], check=True)