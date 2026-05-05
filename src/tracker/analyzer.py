import os
from abc import ABC, abstractmethod
from utils.flow_data import FlowData
from utils.enums import Algorithm

class Analyzer(ABC):
    image_width: int
    image_height: int
    prefixe_file: str
    algorithm: Algorithm
    mask: str
    
    def __init__(self, image_path: str, height: int, width: int, algorithm: Algorithm, mask: str):
        self.image_height = height
        self.image_width = width
        self.algorithm = algorithm
        self.mask = mask
        self.prefixe_file = f"{image_path}_{self.algorithm.value}_{self.mask}_{self.toString()}"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.script_dir, "../../outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def update(self, flow_data: FlowData) -> None:
        ...

    @abstractmethod
    def detectMovements(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def writeData(self, *args, **kwargs) -> None:
        ...
        
    @abstractmethod
    def toString(self) -> str:
        ...