# ============================================================
# CourtVision — Test ensemble XGBoost + CatBoost
# Reproduit l'approche Blackport
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import log_loss
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ── Chargement ───────────────────────────────────────────────
path = kagglehub.dataset_download("mexwell/nba-shots")
df   = pd.read_csv(os.path.join(path, 'NBA_2024_Shots.csv'))
df   = df.rename(columns={'SHOT_MADE': 'SHOT_MADE_FLAG'})
df['SHOT_MADE_FLAG'] = df['SHOT_MADE_FLAG'].astype(int)
df   = df[df['SHOT_DISTANCE'] < 35].copy()
print(f"✅ {len(df):,} tirs charges")

# ── Features ─────────────────────────────────────────────────
df['SHOT_ANGLE']       = np.abs(np.degrees(np.arctan2(df['LOC_X'], df['LOC_Y']))).fillna(0)
df['time_pressure']    = df['MINS_LEFT'] * 60 + df['SECS_LEFT']
df['is_clutch']        = ((df['QUARTER'] >= 4) & (df['time_pressure'] <= 60)).astype(int)
df['shot_distance_sq'] = df['SHOT_DISTANCE'] ** 2
df['is_corner_3']      = ((df['SHOT_TYPE'] == '3PT Field Goal') &
                           (df['LOC_Y'] < 9.2) & (df['LOC_X'].abs() > 22)).astype(int)

def creation_score(action):
    a = str(action).lower()
    if any(k in a for k in ['cutting', 'putback', 'tip', 'running']): return 0
    if any(k in a for k in ['step back', 'fadeaway', 'pullup', 'pull-up']): return 3
    if any(k in a for k in ['driving', 'turnaround']): return 2
    return 1

df['creation_score'] = df['ACTION_TYPE'].apply(creation_score)

FEATURES_NUM = ['SHOT_DISTANCE', 'SHOT_ANGLE', 'time_pressure', 'shot_distance_sq']
FEATURES_CAT = ['creation_score', 'is_corner_3', 'is_clutch']
FEATURES_ALL = FEATURES_NUM + FEATURES_CAT

X = df[FEATURES_ALL]
y = df['SHOT_MADE_FLAG']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── XGBoost seul ─────────────────────────────────────────────
print("\nEntrainement XGBoost...")
pre = ColumnTransformer([('num', StandardScaler(), FEATURES_NUM), ('cat', 'passthrough', FEATURES_CAT)])
pipe_xgb = Pipeline([('pre', pre), ('model', xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic', random_state=42, n_jobs=-1))])
pipe_xgb.fit(X_tr, y_tr)
probas_xgb = pipe_xgb.predict_proba(X_te)[:, 1]
loss_xgb   = log_loss(y_te, probas_xgb)
print(f"  XGBoost seul : {loss_xgb:.4f}")

# ── CatBoost seul ────────────────────────────────────────────
# CatBoost gere nativement les variables categorielles
# et n'a pas besoin de StandardScaler
print("Entrainement CatBoost...")
cat_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    random_seed=42,
    verbose=0,          # silence les logs
)
cat_model.fit(X_tr, y_tr)
probas_cat = cat_model.predict_proba(X_te)[:, 1]
loss_cat   = log_loss(y_te, probas_cat)
print(f"  CatBoost seul : {loss_cat:.4f}")

# ── Ensemble : moyenne des deux probabilites ──────────────────
# Approche Blackport : chaque modele vote, on moyenne
print("Calcul ensemble...")
probas_ensemble = (probas_xgb + probas_cat) / 2
loss_ensemble   = log_loss(y_te, probas_ensemble)
print(f"  Ensemble XGB+Cat : {loss_ensemble:.4f}")

# ── Bilan ────────────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"  BILAN ENSEMBLE")
print(f"{'='*45}")
print(f"  Hasard total       : 0.6931")
print(f"  XGBoost seul       : {loss_xgb:.4f}  (notre modele actuel)")
print(f"  CatBoost seul      : {loss_cat:.4f}")
print(f"  Ensemble XGB+Cat   : {loss_ensemble:.4f}")
print(f"  Gain vs XGBoost    : {'+' if loss_xgb > loss_ensemble else ''}{loss_xgb - loss_ensemble:.4f}")
print(f"  Blackport ref      : 0.6478")
print(f"{'='*45}")