import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .config import FEATURES_ALL, XGB_PARAMS, RANDOM_STATE, CV_FOLDS

def create_pipeline():
    """Crée le pipeline de Machine Learning (XGBoost — pas de scaling, inutile pour les arbres)."""
    pipeline = Pipeline(steps=[
        ('classifier', XGBClassifier(**XGB_PARAMS))
    ])
    return pipeline

def evaluate(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test et retourne Log Loss, ROC AUC et Brier Score."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    loss = log_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    return loss, auc, brier

def cross_validate(model, X, y):
    """Effectue une validation croisée robuste (K-Fold)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # scikit-learn utilise neg_log_loss par défaut (il cherche toujours à maximiser un score)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)
    
    # On repasse les scores en positif pour la lecture
    log_losses = -scores
    return np.mean(log_losses), np.std(log_losses)