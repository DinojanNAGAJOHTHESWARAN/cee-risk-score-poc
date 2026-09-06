import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from src.features.feature_engineering import MODEL_NUMERIC_COLUMNS, MODEL_CATEGORICAL_COLUMNS

class SupervisedRiskModel:
    def __init__(self, algorithm="logistic_regression", min_frequency=8, random_state=42, max_iter=1500, n_estimators=350, min_samples_leaf=4):
        self.algorithm=algorithm
        num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
        cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency))])
        pre=ColumnTransformer([("num",num,MODEL_NUMERIC_COLUMNS),("cat",cat,MODEL_CATEGORICAL_COLUMNS)], remainder="drop")
        if algorithm=="random_forest": clf=RandomForestClassifier(n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,class_weight="balanced_subsample",random_state=random_state,n_jobs=-1)
        else: clf=LogisticRegression(max_iter=max_iter,class_weight="balanced",solver="liblinear",random_state=random_state)
        self.pipeline=Pipeline([("preprocess",pre),("classifier",clf)])
    def fit(self, feature_df, y): self.pipeline.fit(feature_df,y.astype(int)); return self
    def predict_proba(self, feature_df): return self.pipeline.predict_proba(feature_df)[:,1]
    def evaluate(self, feature_df, y, threshold=0.5):
        p=self.predict_proba(feature_df); pred=(p>=threshold).astype(int); y=np.asarray(y,dtype=int)
        return {"roc_auc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),"precision_at_0_5":float(precision_score(y,pred,zero_division=0)),"recall_at_0_5":float(recall_score(y,pred,zero_division=0)),"f1_at_0_5":float(f1_score(y,pred,zero_division=0))}
    def local_logistic_factors(self, one_row, top_n=5):
        clf=self.pipeline.named_steps["classifier"]
        if not hasattr(clf,"coef_"): return []
        pre=self.pipeline.named_steps["preprocess"]; z=pre.transform(one_row)
        if hasattr(z,"toarray"): z=z.toarray()
        vals=np.asarray(z)[0]*clf.coef_[0]; names=pre.get_feature_names_out(); order=np.argsort(np.abs(vals))[::-1][:top_n]
        return [{"feature":str(names[i]),"contribution":float(vals[i])} for i in order if abs(vals[i])>=1e-8]
