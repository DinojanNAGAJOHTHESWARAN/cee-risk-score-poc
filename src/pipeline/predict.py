from pathlib import Path
import argparse
import joblib
from src.config import ROOT
from src.data.loader import load_dataset, validate_schema
from src.models.hybrid_score import score_dataframe

def main(data_path,output_path="data/processed/scored_dossiers.csv",model_path="models/risk_bundle.joblib"):
    bundle=joblib.load(ROOT/model_path); df=load_dataset(data_path); validate_schema(df); scored=score_dataframe(bundle,df); out=ROOT/output_path; out.parent.mkdir(parents=True,exist_ok=True); scored.to_csv(out,index=False); print(f"{len(scored)} dossiers scorés -> {out}")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--output",default="data/processed/scored_dossiers.csv"); args=ap.parse_args(); main(args.data,args.output)
