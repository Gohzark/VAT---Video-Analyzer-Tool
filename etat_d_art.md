<div align="center">

# Etat de l'art

</div>

## 1. Technologies à regarder

### Technos pour masquer le fond d'une image
- MOG2 (Mixture of Gaussians). Pourquoi c'est bien : Si tu as un arbre qui bouge légèrement ou de l'eau qui scintille, le modèle apprend que ces pixels peuvent varier entre "vert clair" et "vert foncé". Il ne les marquera pas comme "mouvement" car cela fait partie du comportement normal du fond.
Application : Idéal pour nettoyer ton flux optique (GMA) avant l'analyse. Tu ne gardes que les vecteurs là où MOG2 détecte un changement.
- MOD (Moving Object Detection) via ViBe. Le concept : Au lieu de faire des stats complexes, il stocke un échantillon de valeurs passées pour chaque pixel. Si la nouvelle valeur est trop différente de son échantillon local, c'est un objet en mouvement.
Avantage : Il "apprend" le fond très vite (dès la deuxième image). Pour un corps qui entre soudainement dans le champ, c'est extrêmement réactif.


### Optical Flow (Flux Optique)
- Lucas-Kanade (sparse)
- Farneback (dense)
- GMA (2021) : Une amélioration de RAFT qui ajoute un mécanisme d'attention (comme dans les Transformers) pour mieux gérer les grandes zones uniformes et les occultations. Permet de récupérer un tenseur (matrice multidimensionnelle qui pour chaque pixel (x,y) associe un vecteur de déplacement (u,v)) pour chaque couple d'images consécutives.
- Megaflow : Grâce à son architecture "Transformer-based" et l'utilisation de DINOv2, les vecteurs sont beaucoup moins "bruités". Là où RAFT pourrait te donner des vecteurs qui tremblent sur un t-shirt uni, MegaFlow rend un mouvement fluide et solidaire pour tout le bras.
Il rend aussi une matrice de probabilité (entre 0 et 1) pour chaque pixel :
    Proche de 1 : Le modèle est sûr de la correspondance.
    Proche de 0 : Le pixel est probablement masqué (occulté) ou est sorti du cadre.
  
### Pose Estimation & Squelettisation
- AlphaPose : Alternative à OpenPose, souvent plus précise sur les postures difficiles (personnes qui se chevauchent, occultations partielles...).
- MMPose (OpenMMLab) — Pas un algorithme unique, mais un framework (boîte à outils) qui regroupe des dizaines de méthodes de pose estimation. Très pratique pour comparer et expérimenter sur Linux.
- Track4World : long-term tracking, retient la position des pixels et les suit au long de la vidéo

### Segmentation Vidéo & Mouvement
- SAM2 (Meta, 2024) — Version vidéo du très célèbre Segment Anything Model. Tu lui montres un objet dans une image, et il l'identifie, le suit et le segmente tout au long de la vidéo. Très émergent et déjà très utilisé en recherche.

### Frameworks & Bibliothèques
- OpenCV — La lib de vision par ordinateur généraliste, incontournable, avec beaucoup d'algos classiques.
- OpenMMLab (MMFlow, MMPose, MMTracking...) — Un écosystème de frameworks modulaires développé par des chercheurs. Chaque "MM" couvre un sous-domaine. Très bien documenté et compatible Linux/PyTorch.
- Supervision (Roboflow) — Utilitaires pratiques pour visualiser, annoter et post-traiter les sorties de tracking/détection

## Technos à ranger
- https://github.com/isl-org/MIDAS (estimation de la profondeur, à combiner avec d'autres techniques)
- https://docs.pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html (réseau deep pour de l'optical flow)
- https://huggingface.co/docs/transformers/model_doc/vit (Vision Transformer, l'équivalent des transformers des LLM mais pour la vidéo)
- https://github.com/xiaofeng94/GMFlowNet (comme Raft)
- https://pq-yang.github.io/projects/MatAnyone2/ (segmentation, nouvelle techno)


