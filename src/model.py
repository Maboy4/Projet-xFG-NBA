from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import xgboost as xgb
from .config import FEATURES_NUM, FEATURES_CAT, XGB_PARAMS, RANDOM_STATE, CV_FOLDS

def create_pipeline(model_type: str = 'xgboost') -> Pipeline:
 preprocessor = ColumnTransformer(transformers=[
 ('num', StandardScaler(), FEATURES_NUM),
 ('cat', 'passthrough', FEATURES_CAT),
 ])
 if model_type == 'xgboost':
    model = xgb.XGBClassifier(**XGB_PARAMS)
 else:
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
 return Pipeline([('preprocessor', preprocessor), ('model', model)])

def evaluate(pipeline, X_test, y_test) -> dict:
 y_proba = pipeline.predict_proba(X_test)[:, 1]
 return {'log_loss': log_loss(y_test, y_proba),
 'roc_auc': roc_auc_score(y_test, y_proba)}

def cross_validate(pipeline, X, y) -> dict:
 skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
 scores = -cross_val_score(pipeline, X, y, cv=skf, scoring='neg_log_loss', n_jobs=-1)
 return {'mean': scores.mean(), 'std': scores.std()}
