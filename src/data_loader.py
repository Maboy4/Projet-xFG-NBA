import os
import pandas as pd
import kagglehub
from .config import MAX_SHOT_DISTANCE

# Saisons disponibles dans le dataset mexwell/nba-shots (2004 à 2024)
ALL_SEASONS = list(range(2004, 2025))


def load_nba_data(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Charge les tirs NBA pour les saisons demandées.
    Par défaut charge toutes les saisons disponibles (2004-2024).
    Exemple : load_nba_data([2022, 2023, 2024])
    """
    if seasons is None:
        seasons = ALL_SEASONS

    path = kagglehub.dataset_download("mexwell/nba-shots")

    dfs = []
    for season in seasons:
        csv_file = os.path.join(path, f'NBA_{season}_Shots.csv')
        if not os.path.exists(csv_file):
            print(f"  Avertissement : {csv_file} introuvable, saison ignorée.")
            continue
        dfs.append(pd.read_csv(csv_file))

    df = pd.concat(dfs, ignore_index=True)
    print(f"OK {len(df):,} tirs chargés ({len(seasons)} saisons : {seasons[0]}–{seasons[-1]})")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'SHOT_MADE' in df.columns:
        df = df.rename(columns={'SHOT_MADE': 'SHOT_MADE_FLAG'})
        df['SHOT_MADE_FLAG'] = df['SHOT_MADE_FLAG'].astype(int)
    df = df[df['SHOT_DISTANCE'] < MAX_SHOT_DISTANCE].copy()
    return df