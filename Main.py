from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from src.data_loader import load_nba_data, clean_data
from src.features import create_features, compute_player_zone_stats, merge_player_zone_stats
from src.model import create_pipeline, evaluate, cross_validate
from src.metrics import compute_player_stats
from src.config import FEATURES_ALL, RANDOM_STATE, TRAINING_SEASONS
from src.visualisation import plot_loss_comparison


def run():
    # 1. Donnees
    df = load_nba_data(TRAINING_SEASONS)
    df = clean_data(df)
    df = create_features(df)

    # 2. Split sur le df complet (pas juste X) pour gérer le data leakage
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['SHOT_MADE_FLAG'])

    # 3. Stats joueur×zone calculées UNIQUEMENT sur le train, puis mergées sur les deux
    player_zone_stats = compute_player_zone_stats(df_train)
    df_train = merge_player_zone_stats(df_train, player_zone_stats)
    df_test  = merge_player_zone_stats(df_test,  player_zone_stats)

    X_train, y_train = df_train[FEATURES_ALL], df_train['SHOT_MADE_FLAG']
    X_test,  y_test  = df_test[FEATURES_ALL],  df_test['SHOT_MADE_FLAG']

    # 4. Baseline Regression Logistique
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    loss_lr = log_loss(y_test, lr.predict_proba(X_test)[:, 1])
    print(f"Log Loss LR  : {loss_lr:.4f}")

    # 5. Modele XGBoost
    pipe = create_pipeline()
    pipe.fit(X_train, y_train)

    # 6. Resultats
    loss_xgb, auc, brier = evaluate(pipe, X_test, y_test)
    print(f"Log Loss XGB : {loss_xgb:.4f} | ROC AUC : {auc:.4f} | Brier : {brier:.4f}")
    cv_mean, cv_std = cross_validate(pipe, X_train, y_train)
    print(f"CV           : {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 7. Stats joueurs (on merge les stats train dans le df complet pour le rapport)
    df_full = merge_player_zone_stats(df, player_zone_stats)
    df_full['xFG'] = pipe.predict_proba(df_full[FEATURES_ALL])[:, 1]
    stats = compute_player_stats(df_full, pipe)
    stats.to_csv('player_xfg_stats.csv', index=False)
    print(stats.head(10)[['PLAYER_NAME', 'FGA', 'FG_PCT', 'xFG_PCT', 'DIFF']].to_string(index=False))
    print(stats.tail(10)[['PLAYER_NAME', 'FGA', 'FG_PCT', 'xFG_PCT', 'DIFF']].to_string(index=False))

    # 8. Export des tirs avec xFG pour filtrage par zone dans l'API
    df_full[['PLAYER_NAME', 'BASIC_ZONE', 'SHOT_MADE_FLAG', 'xFG', 'LOC_X', 'LOC_Y']].to_csv('shots_data.csv', index=False)
    print(f"shots_data.csv exporté ({len(df_full):,} tirs)")

    # 8. Visualisation comparaison LR vs XGBoost
    plot_loss_comparison(loss_lr, loss_xgb)


if __name__ == '__main__':
    run()