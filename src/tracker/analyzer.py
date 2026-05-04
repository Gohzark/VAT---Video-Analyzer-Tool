from abc import ABC, abstractmethod
from utils.flow_data import FlowData

class Analyzer(ABC):
    image_width: int
    image_height: int
    
    def __init__(self, height, width):
        self.image_height = height
        self.image_width = width

    @abstractmethod
    def update(self, flow_data: FlowData) -> None:
        ...

    @abstractmethod
    def detectMovements(self, *args, **kwargs) -> None:
        ...
