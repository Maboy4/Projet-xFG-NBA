"""
api/main.py — API REST pour le projet xFG (NBA shot quality model).

Lancer depuis la racine du projet avec :
    uvicorn api.main:app --reload
"""

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="xFG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement des données au démarrage ───────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[WARNING] '{path}' introuvable. Lance d'abord Main.py.")
        return pd.DataFrame()

_df       = _load_csv("player_xfg_stats.csv")   # stats globales (min 200 tirs)
_shots_df = _load_csv("shots_data.csv")          # tirs avec xFG par zone

MIN_SHOTS_ZONE = 20   # volume minimum par joueur pour figurer dans un leaderboard zonal


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health() -> dict:
    return {"status": "API xFG Opérationnelle 🏀"}


@app.get("/leaderboard")
def leaderboard(
    limit: int = Query(default=10, ge=1),
    zone: str | None = Query(default=None),
) -> list[dict]:
    """
    Retourne le top *limit* joueurs triés par DIFF (POE).
    - Sans zone : classement global depuis player_xfg_stats.csv.
    - Avec zone  : calcul dynamique sur shots_data.csv filtré par BASIC_ZONE.
    """
    if zone:
        if _shots_df.empty:
            raise HTTPException(status_code=503, detail="shots_data.csv manquant. Lance Main.py.")

        filtered = _shots_df[_shots_df["BASIC_ZONE"] == zone]
        if filtered.empty:
            raise HTTPException(status_code=404, detail=f"Zone '{zone}' introuvable.")

        stats = (
            filtered.groupby("PLAYER_NAME")
            .agg(
                FGM=("SHOT_MADE_FLAG", "sum"),
                FGA=("SHOT_MADE_FLAG", "count"),
                FG_PCT=("SHOT_MADE_FLAG", "mean"),
                xFG_PCT=("xFG", "mean"),
            )
            .reset_index()
        )
        stats["DIFF"]    = ((stats["FG_PCT"] - stats["xFG_PCT"]) * 100).round(1)
        stats["FG_PCT"]  = (stats["FG_PCT"]  * 100).round(1)
        stats["xFG_PCT"] = (stats["xFG_PCT"] * 100).round(1)
        stats = stats[stats["FGA"] >= MIN_SHOTS_ZONE].sort_values("DIFF", ascending=False)
        return stats.head(limit).to_dict(orient="records")

    if _df.empty:
        raise HTTPException(status_code=503, detail="player_xfg_stats.csv manquant. Lance Main.py.")
    return _df.head(limit).to_dict(orient="records")


@app.get("/player/{player_name}")
def get_player(player_name: str) -> dict:
    """Recherche un joueur par son nom (insensible à la casse)."""
    if _df.empty:
        raise HTTPException(status_code=503, detail="player_xfg_stats.csv manquant.")

    mask = _df["PLAYER_NAME"].str.contains(player_name, case=False, na=False)
    results = _df[mask]

    if results.empty:
        raise HTTPException(status_code=404, detail=f"Aucun joueur trouvé pour '{player_name}'.")

    return {"results": results.to_dict(orient="records")}


@app.get("/player/{player_name}/shots")
def get_player_shots(player_name: str) -> dict:
    """
    Retourne pour un joueur (nom exact) :
    - zones : FGA / FGM / FG% par zone (pour la heatmap)
    - shots  : coordonnées et résultat de chaque tir (pour le shot chart)
    """
    if _shots_df.empty:
        raise HTTPException(status_code=503, detail="shots_data.csv manquant. Lance Main.py.")

    has_coords = "LOC_X" in _shots_df.columns and "LOC_Y" in _shots_df.columns
    if not has_coords:
        raise HTTPException(status_code=503, detail="shots_data.csv ne contient pas LOC_X/LOC_Y. Régénère-le avec Main.py.")

    player_df = _shots_df[_shots_df["PLAYER_NAME"] == player_name]

    if player_df.empty:
        raise HTTPException(status_code=404, detail=f"Aucun tir trouvé pour '{player_name}'.")

    # Agrégat par zone
    zone_stats = (
        player_df.groupby("BASIC_ZONE")
        .agg(FGA=("SHOT_MADE_FLAG", "count"), FGM=("SHOT_MADE_FLAG", "sum"))
        .reset_index()
    )
    zone_stats["FG_PCT"] = (zone_stats["FGM"] / zone_stats["FGA"] * 100).round(1)

    # Tirs individuels (coordonnées entières pour alléger le JSON)
    shots_cols = ["LOC_X", "LOC_Y", "SHOT_MADE_FLAG"]
    shots = player_df[shots_cols].copy()
    shots["LOC_X"] = shots["LOC_X"].round().astype(int)
    shots["LOC_Y"] = shots["LOC_Y"].round().astype(int)

    return {
        "player": player_name,
        "zones":  zone_stats.to_dict(orient="records"),
        "shots":  shots.to_dict(orient="records"),
    }
