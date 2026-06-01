"""
Individual Probability of Default (PD) Model

Multiple model options:
- Logistic Regression (interpretable)
- Gradient Boosting (accurate)
- Neural Network (complex patterns)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class IndividualPDModel:
    """
    Individual Probability of Default model.

    Predicts P(Default) for each person based on their features,
    independent of network effects.
    """

    def __init__(
        self,
        model_type: str = 'logistic',
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize PD model.

        Parameters
        ----------
        model_type : str
            'logistic' or 'gradient_boosting'
        feature_columns : list, optional
            Features to use. If None, uses default set.
        """
        self.model_type = model_type
        self.feature_columns = feature_columns or [
            'age', 'income', 'employment_years', 'debt_to_income',
            'num_credit_lines', 'missed_payments', 'credit_utilization',
            'account_age_months'
        ]

        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None

    def fit(
        self,
        persons: pd.DataFrame,
        target_col: str = 'default',
        validation_split: float = 0.2
    ) -> Dict:
        """
        Fit the PD model.

        Parameters
        ----------
        persons : pd.DataFrame
            Person features dataframe
        target_col : str
            Column name for default indicator
        validation_split : float
            Fraction for validation

        Returns
        -------
        metrics : dict
            Training and validation metrics
        """
        # Prepare features
        X = persons[self.feature_columns].copy()
        y = persons[target_col].values

        # Handle missing values
        X = X.fillna(X.median())

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Initialize model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                penalty='l2',
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Fit model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Compute feature importance
        if self.model_type == 'logistic':
            self.feature_importance_ = pd.Series(
                np.abs(self.model.coef_[0]),
                index=self.feature_columns
            ).sort_values(ascending=False)
        else:
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=self.feature_columns
            ).sort_values(ascending=False)

        # Compute metrics
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        val_proba = self.model.predict_proba(X_val_scaled)[:, 1]

        metrics = {
            'train_auc': roc_auc_score(y_train, train_proba),
            'val_auc': roc_auc_score(y_val, val_proba),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'default_rate_train': y_train.mean(),
            'default_rate_val': y_val.mean()
        }

        return metrics

    def predict_proba(self, persons: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default for each person.

        Parameters
        ----------
        persons : pd.DataFrame
            Person features

        Returns
        -------
        proba : np.ndarray
            Array of P(Default) for each person
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = persons[self.feature_columns].copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, persons: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary default status."""
        proba = self.predict_proba(persons)
        return (proba >= threshold).astype(int)

    def get_optimal_threshold(
        self,
        persons: pd.DataFrame,
        target_col: str = 'default',
        metric: str = 'f1'
    ) -> float:
        """Find optimal classification threshold."""
        y_true = persons[target_col].values
        y_proba = self.predict_proba(persons)

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        if metric == 'f1':
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            optimal_idx = np.argmax(f1_scores[:-1])
        elif metric == 'precision':
            optimal_idx = np.argmax(precisions[:-1])
        elif metric == 'recall':
            optimal_idx = np.argmax(recalls[:-1])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return thresholds[optimal_idx]

    def explain_prediction(self, person: pd.Series) -> pd.DataFrame:
        """
        Explain individual prediction with feature contributions.

        Parameters
        ----------
        person : pd.Series
            Single person's features

        Returns
        -------
        explanation : pd.DataFrame
            Feature contributions to the prediction
        """
        if not self.is_fitted or self.model_type != 'logistic':
            raise RuntimeError("Only available for fitted logistic model")

        X = person[self.feature_columns].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Compute contributions
        contributions = X_scaled[0] * self.model.coef_[0]

        explanation = pd.DataFrame({
            'feature': self.feature_columns,
            'value': person[self.feature_columns].values,
            'scaled_value': X_scaled[0],
            'coefficient': self.model.coef_[0],
            'contribution': contributions
        })

        explanation['direction'] = np.where(
            explanation['contribution'] > 0,
            'increases_risk',
            'decreases_risk'
        )

        return explanation.sort_values('contribution', ascending=False)


class PDModelEnsemble:
    """
    Ensemble of PD models for robust predictions.
    """

    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models = []

    def fit(self, persons: pd.DataFrame, target_col: str = 'default'):
        """Fit ensemble with bootstrap samples."""
        n = len(persons)

        for i in range(self.n_models):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            sample = persons.iloc[indices]

            # Alternate model types
            model_type = 'logistic' if i % 2 == 0 else 'gradient_boosting'
            model = IndividualPDModel(model_type=model_type)
            model.fit(sample, target_col)
            self.models.append(model)

    def predict_proba(self, persons: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty.

        Returns
        -------
        mean_proba : np.ndarray
            Mean probability across models
        std_proba : np.ndarray
            Standard deviation (uncertainty)
        """
        predictions = np.array([m.predict_proba(persons) for m in self.models])
        return predictions.mean(axis=0), predictions.std(axis=0)


if __name__ == '__main__':
    from data_generator import generate_network

    print("Generating data...")
    persons, _ = generate_network(seed=42)

    print("\nFitting logistic model...")
    model_lr = IndividualPDModel(model_type='logistic')
    metrics_lr = model_lr.fit(persons)
    print(f"Logistic - Train AUC: {metrics_lr['train_auc']:.3f}, Val AUC: {metrics_lr['val_auc']:.3f}")

    print("\nFitting gradient boosting model...")
    model_gb = IndividualPDModel(model_type='gradient_boosting')
    metrics_gb = model_gb.fit(persons)
    print(f"GBM - Train AUC: {metrics_gb['train_auc']:.3f}, Val AUC: {metrics_gb['val_auc']:.3f}")

    print("\nFeature importance (Logistic):")
    print(model_lr.feature_importance_)

    print("\nSample prediction explanation:")
    explanation = model_lr.explain_prediction(persons.iloc[0])
    print(explanation)

    persons['predicted_pd'] = model_lr.predict_proba(persons)
    print(f"\nPredicted PD range: [{persons['predicted_pd'].min():.3f}, {persons['predicted_pd'].max():.3f}]")
