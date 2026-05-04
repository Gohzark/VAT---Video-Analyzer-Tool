from enum import Enum

class Algorithm(Enum):
    LK = "LK"
    FARNEBACK = "Farneback"

    def __str__(self):
        return self.name
    
class Mask(Enum):
    MO2 = "MO2"
    No = "No"
    
    def __str__(self):
        return self.name
    
class Tracker(Enum):
    Fourier = "Fourier"
    StartStop = "StartStop"
    
    def __str__(self):
        return self.name