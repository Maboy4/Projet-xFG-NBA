import os
import pandas as pd
import kagglehub
from .config import MAX_SHOT_DISTANCE
def load_nba_data() -> pd.DataFrame:
 path = kagglehub.dataset_download("mexwell/nba-shots")
 csv_file = os.path.join(path, 'NBA_2024_Shots.csv')
 df = pd.read_csv(csv_file)
 print(f"OK {len(df):,} tirs charges")
 return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
 df = df.copy()
 if 'SHOT_MADE' in df.columns:
    df = df.rename(columns={'SHOT_MADE': 'SHOT_MADE_FLAG'})
    df['SHOT_MADE_FLAG'] = df['SHOT_MADE_FLAG'].astype(int)
 df = df[df['SHOT_DISTANCE'] < MAX_SHOT_DISTANCE].copy()
 return df