04/05/2026 : 

    Dans un premier temps, pour extraire un rythme de mouvements réguliers à partir d'une vidéo de sport, j'ai déterminer des points à traquer sur l'image avec l'algorithme Shi Tomasi, grâce à la fonction cv.goodFeaturesToTrack de OpenCV. C'est un algorithme de détections de coins de 1994. Il se base sur la détection d'un fort gradient (une forte variation d'intensité lumineuse) dans toutes les directions (horizontale et verticale) pour considérer que c'est un coin.
    Ensuite, pour traquer ces points sur l'image suivante, j'ai utiliser Lucas-Kanade avec la méthode pyramidale (1981) grâce à la fonction cv.calcOpticalFlowPyrLK de OpenCV. C'est un algorithme sparse donc qui suit seulement certains points déterminés sur l'image, ce qui lui confère l'avantage d'être rapide à exécuter.
    Ainsi, on récupère le flux optique entre chaque paire d'images. C'est-à-dire un ensemble de vecteurs de déplacement de chaque point traqué par Lucas-Kanade entre les 2 images.

    Après avoir récupéré ce signal, il fallait l'étudier pour en extraire un rythme.
    
    Ma première approche a été d'observer la vidéo que je tentais d'analyser pour trouver une façon de procéder.
    Il s'agissait d'une vidéo d'un individu effectuer des tractions au poids du corps en suivant un rythme régulier, et marquant un temps d'arrêt en bas et en haut du mouvement.
    Je suis donc parti de la supposition qu'en parvenant à detecter les changements entre les moments où l'individu effectue le mouvement et ceux où il marque une pause, je pourrais compter le nombre de répétitions effectué et extraire un rythme.

    J'ai alors créer un Anal


