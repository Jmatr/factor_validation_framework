### 4. 因子基类 (factors/factor_base.py)

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Factor(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
    def calculate(self, data):
        """Calculate factor values"""
        pass

    def __str__(self):
        return f"Factor: {self.name} - {self.description}"