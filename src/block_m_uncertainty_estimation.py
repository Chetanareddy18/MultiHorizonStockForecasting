# ==========================================================
# BLOCK M: UNCERTAINTY ESTIMATION MODULE
# ==========================================================
# Provides probabilistic forecasting via:
# 1. Monte Carlo Dropout
# 2. Quantile Regression
# 3. Bootstrapping
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import QuantileRegressor
from sklearn.utils import resample


# ==========================================================
# 1️⃣ MONTE CARLO DROPOUT
# ==========================================================

class MCDropoutPredictor:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def predict(self, X, n_samples=100):
        self.model.train()  # IMPORTANT: keep dropout active

        preds = []

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(X_tensor)
                preds.append(output.cpu().numpy())

        preds = np.array(preds)

        mean_prediction = preds.mean(axis=0)
        std_prediction = preds.std(axis=0)

        lower = mean_prediction - 1.96 * std_prediction
        upper = mean_prediction + 1.96 * std_prediction

        return {
            "mean": mean_prediction,
            "std": std_prediction,
            "lower_95": lower,
            "upper_95": upper
        }


# ==========================================================
# 2️⃣ QUANTILE REGRESSION
# ==========================================================

class QuantileRegressionPredictor:
    def __init__(self, quantiles=[0.05, 0.5, 0.95], alpha=0.0001):
        self.quantiles = quantiles
        self.models = {
            q: QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
            for q in quantiles
        }

    def fit(self, X, y):
        for q, model in self.models.items():
            model.fit(X, y)

    def predict(self, X):
        predictions = {}
        for q, model in self.models.items():
            predictions[f"quantile_{int(q*100)}"] = model.predict(X)

        return predictions


# ==========================================================
# 3️⃣ BOOTSTRAPPING
# ==========================================================

class BootstrapPredictor:
    def __init__(self, base_model, n_bootstrap=100):
        self.base_model = base_model
        self.n_bootstrap = n_bootstrap

    def predict(self, X, y_train, X_pred):
        preds = []

        for _ in range(self.n_bootstrap):
            X_resampled, y_resampled = resample(X, y_train)

            model = self.base_model
            model.fit(X_resampled, y_resampled)

            pred = model.predict(X_pred)
            preds.append(pred)

        preds = np.array(preds)

        mean_prediction = preds.mean(axis=0)
        std_prediction = preds.std(axis=0)

        lower = np.percentile(preds, 5, axis=0)
        upper = np.percentile(preds, 95, axis=0)

        return {
            "mean": mean_prediction,
            "std": std_prediction,
            "lower_90": lower,
            "upper_90": upper
        }