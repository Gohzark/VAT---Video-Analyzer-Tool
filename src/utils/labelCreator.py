import cv2
import json
import os
import argparse


# Pour ajouter des labels, télécharger le fichier data.json depuis https://www.kaggle.com/datasets/tinodolbeau/opticalflow-videos.
# Placer le à la racine du projet puis exécuter ce script.
# Ensuite mettez la nouvelle version de data.json sur kaggle.
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    return {
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2)
    }

def update_label_json(json_path, video_name, freq, reps, video_info):
    data = {}
    
    # Charger l'existant si le fichier existe
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    # Ajouter/Mettre à jour l'entrée
    data[video_name] = {
        "ground_truth": {
            "frequency_hz": freq,
            "period_seconds": round(1/freq, 3) if freq > 0 else 0,
            "repetitions": reps
        },
        "technical_info": video_info
    }

    # Sauvegarder
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Étiquette enregistrée pour {video_name} dans {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Créateur de labels pour analyse vidéo")
    parser.add_argument("--video", type=str, required=True, help="Nom ou chemin du fichier vidéo")
    parser.add_argument("--freq", type=float, required=True, help="Fréquence observée (Hz)")
    parser.add_argument("--reps", type=int, required=True, help="Nombre de répétitions comptées")
    parser.add_argument("--output", type=str, default="resources/data.json", help="Fichier JSON de sortie")

    args = parser.parse_args()
    
    # 1. Vérifier si la vidéo existe
    if not os.path.exists(args.video):
        print(f"❌ Fichier vidéo introuvable : {args.video}")
    else:
        # 2. Extraire les infos auto
        info = get_video_info(args.video)
        
        if info:
            # 3. Enregistrer dans le JSON
            video_filename = os.path.basename(args.video)
            update_label_json(args.output, video_filename, args.freq, args.reps, info)