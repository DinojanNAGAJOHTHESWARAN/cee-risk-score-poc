import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ANOMALY_COLUMNS=["VolumeCEEDeclare","VolumeCEECalculeAttribue","VolumeGapRel","VolumeRatio","DaysOperationToSubmission","PartnerFrequency","FicheFrequency","CommuneFrequency","ComboFrequency","FicheVolumeRatioToMedian"]

class AnomalyRiskModel:
    def __init__(self,n_estimators=250,contamination="auto",random_state=42):
        self.pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",RobustScaler()),("model",IsolationForest(n_estimators=n_estimators,contamination=contamination,random_state=random_state,n_jobs=-1))]); self.reference_scores=None
    def fit(self,feature_df):
        x=feature_df[ANOMALY_COLUMNS]; self.pipe.fit(x); self.reference_scores=np.sort(-self.pipe.decision_function(x)); return self
    def predict_risk(self,feature_df):
        raw=-self.pipe.decision_function(feature_df[ANOMALY_COLUMNS]); n=max(len(self.reference_scores),1); ranks=np.searchsorted(self.reference_scores,raw,side="right"); return np.clip(ranks/n*100.0,0,100)
