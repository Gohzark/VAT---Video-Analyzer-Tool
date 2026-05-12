from enum import Enum

class Algorithm(Enum):
    LucasKanade = "LK"
    Farneback = "Farneback"

    def __str__(self):
        return self.name
    
class Mask(Enum):
    MOG2 = "MOG2"
    NoMask = "NoMask"
    
    def __str__(self):
        return self.name
    
class Centering(Enum):
    ExponentialMovingAverage = "EMA"
    NoCentering = "NoCentering"
    Kalman = "Kalman"
    def __str__(self):
        return self.name
    

class Analyze(Enum):
    FastFourierTransformation = "FFT"
    StartStop = "SS"
    Sliding = "Sliding"
    
    def __str__(self):
        return self.name
    