import numpy as np
import pandas as pd

SMOOTHING = 10  # facteur bayésien : plus c'est élevé, plus on tire vers la moyenne globale


def compute_player_zone_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le taux de réussite historique par joueur × zone de tir.
    DOIT être appelé uniquement sur les données d'entraînement pour éviter le data leakage.
    Utilise un lissage bayésien pour les joueurs avec peu de tirs dans une zone.
    """
    global_mean = df_train['SHOT_MADE_FLAG'].mean()

    stats = df_train.groupby(['PLAYER_NAME', 'BASIC_ZONE']).agg(
        fgm=('SHOT_MADE_FLAG', 'sum'),
        fga=('SHOT_MADE_FLAG', 'count'),
    ).reset_index()

    # Lissage : (fgm + moyenne_globale * k) / (fga + k)
    # Un joueur avec 2 tirs dans une zone ne donnera pas un taux de 100%
    stats['player_zone_xfg'] = (
        (stats['fgm'] + global_mean * SMOOTHING) / (stats['fga'] + SMOOTHING)
    )

    return stats[['PLAYER_NAME', 'BASIC_ZONE', 'player_zone_xfg']]


def merge_player_zone_stats(df: pd.DataFrame, player_zone_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge les stats joueur×zone dans df.
    Les combinaisons inconnues (nouveau joueur ou zone rare) reçoivent la moyenne globale.
    """
    fallback = player_zone_stats['player_zone_xfg'].mean()
    df = df.merge(player_zone_stats, on=['PLAYER_NAME', 'BASIC_ZONE'], how='left')
    df['player_zone_xfg'] = df['player_zone_xfg'].fillna(fallback)
    return df


# Ordre naturel distance → entier croissant
_ZONE_RANGE_MAP = {
    'Less Than 8 ft.': 0,
    '8-16 ft.':        1,
    '16-24 ft.':       2,
    '24+ ft.':         3,
    'Back Court Shot': 4,
}

# Zones ordonnées du plus facile au plus difficile
_BASIC_ZONE_MAP = {
    'Restricted Area':        0,
    'In The Paint (Non-RA)':  1,
    'Mid-Range':              2,
    'Left Corner 3':          3,
    'Right Corner 3':         4,
    'Above the Break 3':      5,
    'Backcourt':              6,
}

_POSITION_MAP = {'G': 0, 'F': 1, 'C': 2}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Angle de tir
    df['SHOT_ANGLE'] = np.abs(np.degrees(np.arctan2(df['LOC_X'], df['LOC_Y']))).fillna(0)

    # Score de création (difficulté du geste)
    conditions = [
        df['ACTION_TYPE'].str.contains('Step Back|Pullup|Fadeaway', case=False, na=False),
        df['ACTION_TYPE'].str.contains('Driving|Drive', case=False, na=False),
        df['ACTION_TYPE'].str.contains('Jump', case=False, na=False),
        df['ACTION_TYPE'].str.contains('Cutting|Cut|Dunk|Alley Oop|Layup', case=False, na=False),
    ]
    df['creation_score'] = np.select(conditions, [3, 2, 1, 0], default=1)

    # Features existantes
    df['is_corner_3'] = ((df['SHOT_TYPE'] == '3PT Field Goal') &
                         (df['LOC_Y'] < 9.2) & (df['LOC_X'].abs() > 22)).astype(int)
    df['time_pressure'] = df['MINS_LEFT'] * 60 + df['SECS_LEFT']
    df['is_clutch'] = ((df['QUARTER'] >= 4) & (df['time_pressure'] <= 60)).astype(int)
    df['shot_distance_sq'] = df['SHOT_DISTANCE'] ** 2

    # Nouvelles features : encodage des zones et position
    df['zone_basic_enc'] = df['BASIC_ZONE'].map(_BASIC_ZONE_MAP).fillna(2).astype(int)
    df['zone_range_enc'] = df['ZONE_RANGE'].map(_ZONE_RANGE_MAP).fillna(2).astype(int)
    df['position_enc']   = df['POSITION_GROUP'].map(_POSITION_MAP).fillna(1).astype(int)

    return df