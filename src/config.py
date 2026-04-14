MAX_SHOT_DISTANCE = 35

# Saisons utilisées pour l'entraînement (2023-24 optimal — au-delà, l'évolution du jeu dégrade le modèle)
TRAINING_SEASONS = [2022, 2023, 2024]

FEATURES_ALL = [
    'SHOT_DISTANCE',
    'SHOT_ANGLE',
    'time_pressure',
    'shot_distance_sq',
    'is_corner_3',
    'is_clutch',
    'creation_score',
    'QUARTER',
    'zone_basic_enc',   # zone de tir (Restricted Area, Mid-Range, Corner 3, etc.)
    'zone_range_enc',   # tranche de distance (<8ft, 8-16, 16-24, 24+)
    'position_enc',     # position du joueur (G / F / C)
    # 'player_zone_xfg' : retire — ajoute du bruit sur une seule saison (zone_basic_enc couvre déjà le signal)
]

MIN_SHOTS_PLAYER = 200
RANDOM_STATE = 42
CV_FOLDS = 5

XGB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.02012574620277875,
    'max_depth': 5,
    'subsample': 0.9869937893814524,
    'colsample_bytree': 0.6165292543805662,
     
    'eval_metric': 'logloss'
}