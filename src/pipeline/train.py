from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
from src.config import ROOT, load_config
from src.data.loader import load_dataset, validate_schema, build_target
from src.features.feature_engineering import FeatureContext, parse_mixed_date
from src.models.supervised import SupervisedRiskModel
from src.models.anomaly import AnomalyRiskModel
from src.models.business_rules import business_scores

def temporal_split(df,y,test_fraction=0.25):
    dates=parse_mixed_date(df["DateCreationSoumissionDossier"]); order=np.argsort(dates.fillna(pd.Timestamp("1900-01-01")).values); cut=max(1,int(len(order)*(1-test_fraction))); tr=order[:cut]; te=order[cut:]; return df.iloc[tr].copy(),df.iloc[te].copy(),y.iloc[tr].copy(),y.iloc[te].copy()

def evaluate_scores(y,scores,threshold=50):
    p=np.asarray(scores)/100.0; pred=(np.asarray(scores)>=threshold).astype(int); y=np.asarray(y,dtype=int); precision,recall,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    return {"roc_auc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),"threshold":float(threshold),"precision":float(precision),"recall":float(recall),"f1":float(f1),"confusion_matrix":confusion_matrix(y,pred).tolist()}

def top_k_metrics(y,scores,fractions=(0.10,0.20)):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float); order=np.argsort(scores)[::-1]; total_pos=max(1,int(y.sum())); out={}
    for frac in fractions:
        n=max(1,int(np.ceil(len(y)*frac))); selected=y[order[:n]]; tp=int(selected.sum()); out[f"Top {int(frac*100)}%"]={"selected_rows":int(n),"recall":float(tp/total_pos),"precision":float(tp/n)}
    return out

def main(data_path=None,config_path=None):
    cfg=load_config(config_path); data_path=Path(data_path or ROOT/cfg["data"]["default_path"]); df=load_dataset(data_path,cfg["data"].get("sheet_name",0)); validate_schema(df,require_target=True); y_all=build_target(df); labeled=y_all.notna(); labeled_df=df.loc[labeled].copy(); y=y_all.loc[labeled].astype(int); train_df,test_df,y_train,y_test=temporal_split(labeled_df,y,cfg["model"]["test_fraction"])
    cutoff=parse_mixed_date(train_df["DateCreationSoumissionDossier"]).max(); all_dates=parse_mixed_date(df["DateCreationSoumissionDossier"]); history_df=df.loc[all_dates<=cutoff].copy(); context=FeatureContext(alpha=cfg["business_rules"]["smoothing_alpha"],use_calculated_volume_rules=cfg["business_rules"].get("use_calculated_volume_rules",False)).fit(history_df,train_df,y_train); x_train=context.transform(train_df); x_test=context.transform(test_df); x_history=context.transform(history_df)
    mcfg=cfg["model"]; sup=SupervisedRiskModel(algorithm=mcfg["supervised_algorithm"],min_frequency=mcfg["one_hot_min_frequency"],random_state=cfg["project"]["random_state"],max_iter=mcfg["max_iter"],n_estimators=mcfg["n_estimators"],min_samples_leaf=mcfg["min_samples_leaf"]).fit(x_train,y_train); acfg=cfg["anomaly"]; ano=AnomalyRiskModel(n_estimators=acfg["n_estimators"],contamination=acfg["contamination"],random_state=cfg["project"]["random_state"]).fit(x_history)
    sup_test=sup.predict_proba(x_test)*100; ano_test=ano.predict_risk(x_test); biz_test,_=business_scores(x_test,context); hcfg=cfg["hybrid_score"]; weights={"supervised":float(hcfg["supervised_weight"]),"business":float(hcfg["business_weight"]),"anomaly":float(hcfg["anomaly_weight"])}; hybrid=weights["supervised"]*sup_test+weights["business"]*np.asarray(biz_test)+weights["anomaly"]*ano_test
    metrics={"dataset_rows":int(len(df)),"labeled_rows":int(len(labeled_df)),"train_rows":int(len(train_df)),"test_rows":int(len(test_df)),"positive_rate_train":float(y_train.mean()),"positive_rate_test":float(y_test.mean()),"temporal_cutoff":str(cutoff),"supervised":sup.evaluate(x_test,y_test),"hybrid":evaluate_scores(y_test,hybrid,50),"top_k":top_k_metrics(y_test,hybrid),"warning":"Le score estime un risque de contrôle défavorable/non-conformité; il ne constitue pas une preuve de fraude."}
    bundle={"version":"0.2.0","context":context,"supervised":sup,"anomaly":ano,"weights":weights,"medium_threshold":int(hcfg["medium_threshold"]),"high_threshold":int(hcfg["high_threshold"]),"metrics":metrics}; model_dir=ROOT/"models"; model_dir.mkdir(exist_ok=True); joblib.dump(bundle,model_dir/"risk_bundle.joblib"); (model_dir/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default=None); ap.add_argument("--config",default=None); args=ap.parse_args(); main(args.data,args.config)
