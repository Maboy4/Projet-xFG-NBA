import numpy as np
import pandas as pd
from .config import CREATION_KEYWORDS 
def create_features(df: pd.DataFrame) -> pd.DataFrame:
 df = df.copy()
 df['SHOT_ANGLE'] = np.abs(np.degrees(np.arctan2(df['LOC_X'], df['LOC_Y']))).fillna(0)
 df['is_creation'] = df['ACTION_TYPE'].apply(_is_creation)
 df['is_corner_3'] = ((df['SHOT_TYPE'] == '3PT Field Goal') &
 (df['LOC_Y'] < 9.2) & (df['LOC_X'].abs() > 22)).astype(int)
 df['time_pressure'] = df['MINS_LEFT'] * 60 + df['SECS_LEFT']
 df['is_clutch'] = ((df['QUARTER'] >= 4) & (df['time_pressure'] <= 60)).astype(int)
 df['shot_distance_sq'] = df['SHOT_DISTANCE'] ** 2
 return df

def _is_creation(action_type: str) -> int:
 """Fonction privee (prefixe _) : non importee depuis l'exterieur."""
 a = str(action_type).lower()
 return 1 if any(k.lower() in a for k in CREATION_KEYWORDS) else 0