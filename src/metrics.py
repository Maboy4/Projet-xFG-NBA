import pandas as pd
from .config import FEATURES_ALL, MIN_SHOTS_PLAYER


def compute_player_stats(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURES_ALL]
    df['xFG'] = pipeline.predict_proba(X)[:, 1]
    stats = df.groupby('PLAYER_NAME').agg(
        FGM=('SHOT_MADE_FLAG', 'sum'),
        FGA=('SHOT_MADE_FLAG', 'count'),
        FG_PCT=('SHOT_MADE_FLAG', 'mean'),
        xFG_PCT=('xFG', 'mean'),
    ).reset_index()
    stats['DIFF'] = ((stats['FG_PCT'] - stats['xFG_PCT']) * 100).round(1)
    stats['FG_PCT'] = (stats['FG_PCT'] * 100).round(1)
    stats['xFG_PCT'] = (stats['xFG_PCT'] * 100).round(1)
    return stats[stats['FGA'] >= MIN_SHOTS_PLAYER].sort_values('DIFF', ascending=False)
