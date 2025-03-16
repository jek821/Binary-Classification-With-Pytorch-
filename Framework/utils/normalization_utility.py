import numpy as np
import pandas as pd

class Normalization:
    @staticmethod
    def min_max_normalize(data):
        """Normalizes the given DataFrame using min-max scaling to range [0, 1]."""
        return (data - data.min()) / (data.max() - data.min())
    
    @staticmethod
    def z_score_normalize(data):
        """Normalizes the given DataFrame using z-score normalization (mean 0, std 1)."""
        return (data - data.mean()) / data.std()