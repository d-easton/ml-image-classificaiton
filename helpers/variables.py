from enum import Enum

class ModelSize(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

class Dataset:
    """
    Models a dataset by associating supported Dataset type with a description
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description