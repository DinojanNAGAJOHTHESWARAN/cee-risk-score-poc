import pandas as pd
from src.data.loader import build_target, POST_CONTROL_COLUMNS
from src.data.validation import ingestion_report, sanitize_for_scoring
from src.models.hybrid_score import risk_level

def test_target_taxonomy():
    df=pd.DataFrame({"StatutFinalControle":["Satisfaisant","Refusé","Inaccessible / non vérifiable"]}); y=build_target(df); assert y.iloc[0]==0; assert y.iloc[1]==1; assert pd.isna(y.iloc[2])

def test_risk_levels():
    assert risk_level(10)=="Faible"; assert risk_level(40)=="Modéré"; assert risk_level(70)=="Élevé"

def test_post_control_removed():
    df=pd.DataFrame({"ReferenceFicheCEE":["X"],"StatutFinalControle":["Refusé"]}); clean=sanitize_for_scoring(df); assert "StatutFinalControle" not in clean.columns

def test_post_control_catalogue():
    assert "ResultatControle" in POST_CONTROL_COLUMNS; assert "MotifRejet" in POST_CONTROL_COLUMNS

def test_ingestion_detects_missing_required():
    r=ingestion_report(pd.DataFrame({"ReferenceFicheCEE":["X"]})); assert not r["can_score"]; assert len(r["missing_required"])>0

def test_ingestion_detects_post_control():
    df=pd.DataFrame({"ReferenceFicheCEE":["X"],"StatutFinalControle":["Satisfaisant"]}); r=ingestion_report(df); assert "StatutFinalControle" in r["post_control_columns"]
