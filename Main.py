from sklearn.model_selection import train_test_split
from src.data_loader import load_nba_data, clean_data
from src.features import create_features
from src.model import create_pipeline, evaluate, cross_validate
from src.metrics import compute_player_stats
from src.config import FEATURES_ALL, RANDOM_STATE
from src.visualisation import plot_loss_comparison


def run():
    # 1. Donnees
    df = load_nba_data()
    df = clean_data(df)
    df = create_features(df)

    # 2. Split
    X = df[FEATURES_ALL]
    y = df['SHOT_MADE_FLAG']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 3. Modele
    pipe = create_pipeline('xgboost')
    pipe.fit(X_train, y_train)

    # 4. Resultats
    m  = evaluate(pipe, X_test, y_test)
    cv = cross_validate(pipe, X, y)
    print(f"Log Loss : {m['log_loss']:.4f} | ROC AUC : {m['roc_auc']:.4f}")
    print(f"CV       : {cv['mean']:.4f} (+/- {cv['std']*2:.4f})")

    # 5. Stats joueurs
    stats = compute_player_stats(df, pipe)
    stats.to_csv('player_xfg_stats.csv', index=False)
    print(stats.head(10)[['PLAYER_NAME', 'FGA', 'FG_PCT', 'xFG_PCT', 'DIFF']].to_string(index=False))

    # 6. Visualisation
    plot_loss_comparison(m['log_loss'], m['log_loss'])


if __name__ == '__main__':
    run()