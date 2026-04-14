import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.data_loader import load_nba_data, clean_data
from src.features import create_features
from src.config import FEATURES_ALL, RANDOM_STATE, MIN_SHOTS_PLAYER, TRAINING_SEASONS

def make_objective(X, y):
    """Retourne la fonction objective avec X et y capturés en closure."""
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': RANDOM_STATE,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**param)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)
        return -scores.mean()
    return objective

if __name__ == "__main__":
    print("Chargement et nettoyage des données...")

    df = load_nba_data(TRAINING_SEASONS)
    df = clean_data(df)
    df = create_features(df)

    player_counts = df['PLAYER_NAME'].value_counts()
    valid_players = player_counts[player_counts >= MIN_SHOTS_PLAYER].index
    df_filtered = df[df['PLAYER_NAME'].isin(valid_players)].copy()

    X = df_filtered[FEATURES_ALL]
    y = df_filtered['SHOT_MADE_FLAG']

    print("Lancement de l'optimisation Bayésienne (20 essais)...")

    study = optuna.create_study(direction='minimize')
    study.optimize(make_objective(X, y), n_trials=20)

    print("\nMeilleurs paramètres trouvés :")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")

    print(f"\nMeilleur Log Loss obtenu : {study.best_value:.4f}")