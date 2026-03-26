MAX_SHOT_DISTANCE = 35
FEATURES_NUM = [
 'SHOT_DISTANCE',
 'SHOT_ANGLE',
 'time_pressure',
 'shot_distance_sq',
]
FEATURES_CAT = ['is_creation', 'is_corner_3', 'is_clutch']
FEATURES_ALL = FEATURES_NUM + FEATURES_CAT
CREATION_KEYWORDS = ['Pullup', 'Pull-Up', 'Step Back', 'Fadeaway', 'Turnaround', 'Driving']
XGB_PARAMS = {
 'n_estimators': 200,
 'max_depth': 6,
 'learning_rate': 0.1,
 'subsample': 0.8,
 'colsample_bytree': 0.8,
 'objective': 'binary:logistic',
 'eval_metric': 'logloss',
 'random_state': 42,
 'n_jobs': -1,
}

MIN_SHOTS_PLAYER = 200
RANDOM_STATE = 42
CV_FOLDS = 5