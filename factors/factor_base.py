from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Factor(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}

    @abstractmethod
    def calculate(self, data: dict) -> pd.DataFrame:
        """Calculate factor values"""
        pass

    def validate_data(self, data: dict, required_fields: list) -> bool:
        """Validate if required data fields are available"""
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required data field '{field}' not available")
        return True

    def normalize_factor(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        """Normalize factor values using z-score"""
        return (factor_values - factor_values.mean()) / factor_values.std()

    def winsorize_factor(self, factor_values: pd.DataFrame, limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
        """Winsorize factor values to handle outliers"""
        quantiles = factor_values.quantile(limits)
        return factor_values.clip(lower=quantiles.iloc[0], upper=quantiles.iloc[1], axis=1)

    def __str__(self):
        return f"Factor: {self.name} - {self.description}"

    def __repr__(self):
        return f"Factor(name='{self.name}', description='{self.description}')"