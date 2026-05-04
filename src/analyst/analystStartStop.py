import numpy as np
from scipy.ndimage import minimum_filter
from analyst.analyst import Analyst

class AnalystStartStop(Analyst):
    means_x: list[float]
    means_y: list[float]
    threshold: float
    
    def __init__(self, height, width, threshold=0.2):
        super().__init__(height, width)
        self.means_x = []
        self.means_y = []
        self.threshold = threshold
        
    def update(self, currents_points, old_points):
        distances = np.linalg.norm(currents_points - old_points, axis=1)
        #supprime le bruit des petits mouvements grace au seuil
        moving_points = currents_points[distances > self.threshold]
        if len(moving_points) > 0:
            means_xy = np.mean(moving_points, axis=0)
            self.means_x.append(means_xy[0])
            self.means_y.append(self.image_height - means_xy[1])
        else:
            if self.means_x:
                self.means_x.append(self.means_x[-1])
                self.means_y.append(self.means_y[-1])
            else:
                self.means_x.append(self.image_width / 2)
                self.means_y.append(self.image_height / 2)
        
    
    #Pas terrible, détecte les arrêts et les départs de mouvements, mais pas les mouvements continus.
    def detectMovements(self):
        #Pour chaque paire de points consécutifs, on calcule le déplacement en X et en Y, puis on les regroupe en une matrice (N-1, 2).
        diffs_x = np.diff(self.means_x)       
        diffs_y = np.diff(self.means_y)       
        matrix_diffs = np.column_stack((diffs_x, diffs_y))
        #On calcule la norme euclidienne de chaque vecteur déplacement.
        norms = np.linalg.norm(matrix_diffs, axis=1)
        #On applique un filtre minimum sur une fenêtre de 5 valeurs. Ça supprime les petits bruits.
        clean_window = minimum_filter(norms, size=5, mode='nearest')
        #Crée un vecteur ou chaque valeur de clean_window supérieure à 0 devient True.
        vecteur_bool = (clean_window > 0)
        #np.diff sur le vecteur de bool donne +1 à chaque début de mouvement et -1 à chaque fin. 
        changes = np.diff(vecteur_bool.astype(int))
        print(changes)     
        nb_movement = np.sum(changes == 1)
        indices_start_movements = np.argwhere(changes==1)
        indices_end_movements = np.argwhere(changes==-1)
        start_serie = indices_start_movements[0].item()
        end_serie = indices_end_movements[-1].item()
        #Si le premier élément est déjà 1, le mouvement est déja en cours donc on ajoute à la main.
        if len(vecteur_bool) > 0 and vecteur_bool[0]:
            nb_movement += 1
        return round(nb_movement, 2), start_serie, end_serie

        
    def getRythm(self, nbMovements, nbFrame, frameRate):
        time = nbFrame / frameRate
        rythm = nbMovements / time
        return round(rythm, 2)
        
        
        