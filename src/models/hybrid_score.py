import numpy as np
import pandas as pd
from .business_rules import business_scores

def risk_level(score,medium=40,high=70):
    if score>=high:return "Élevé"
    if score>=medium:return "Modéré"
    return "Faible"

def score_dataframe(bundle,raw_df):
    context=bundle["context"]; features=context.transform(raw_df); sup=bundle["supervised"].predict_proba(features)*100; ano=bundle["anomaly"].predict_risk(features); biz,signals=business_scores(features,context); w=bundle["weights"]
    hybrid=np.clip(w["supervised"]*sup+w["business"]*np.asarray(biz)+w["anomaly"]*ano,0,100); medium=bundle.get("medium_threshold",40); high=bundle.get("high_threshold",70)
    result=raw_df.copy(); result["_row_id"]=np.arange(len(result)); result["ScoreSupervise"]=np.round(sup,2); result["ScoreMetier"]=np.round(biz,2); result["ScoreAnomalie"]=np.round(ano,2); result["ScoreRisque"]=np.round(hybrid,2); result["NiveauRisque"]=[risk_level(s,medium,high) for s in hybrid]; result["Signaux"]=signals
    result["RaisonsMetier"]=[" | ".join(sig["title"] for sig in ss if sig["kind"]!="uncertainty") if any(sig["kind"]!="uncertainty" for sig in ss) else "Aucun signal métier fort" for ss in signals]
    return result.sort_values("ScoreRisque",ascending=False)

def _friendly_feature(name):
    mapping={"num__ComboFrequency":"Rareté de la combinaison fiche / bâtiment","num__DaysOperationToSubmission":"Délai entre l'opération et la soumission","num__VolumeCEEDeclare":"Volume CEE déclaré","num__PartnerFrequency":"Nombre de dossiers historiques du partenaire","num__FicheFrequency":"Fréquence historique de la fiche CEE","num__CommuneFrequency":"Fréquence historique dans la commune","num__FicheVolumeRatioToMedian":"Volume comparé à la médiane de la fiche","num__SubmissionMonth":"Mois de soumission","num__SubmissionDayOfWeek":"Jour de la semaine de soumission"}
    if name in mapping:return mapping[name]
    if name.startswith("cat__"):
        raw=name[5:]; prefixes={"ReferenceFicheCEE_":"Fiche CEE : ","SecteurOperation_":"Secteur : ","DomaineOperation_":"Domaine : ","TypeBeneficiaire_":"Bénéficiaire : ","TypeBatiment_":"Bâtiment : ","Region_":"Région : ","Departement_":"Département : ","EquipmentBrand_":"Marque équipement : "}
        for prefix,label in prefixes.items():
            if raw.startswith(prefix):return label+raw[len(prefix):]
        return raw.replace("_"," ")
    return name.replace("num__","").replace("_"," ")

def explain_one(bundle,raw_row):
    context=bundle["context"]; feat=context.transform(raw_row); scored=score_dataframe(bundle,raw_row).iloc[0]; factors=bundle["supervised"].local_logistic_factors(feat,top_n=6)
    for f in factors:f["label"]=_friendly_feature(f["feature"]); f["direction"]="augmente" if f["contribution"]>0 else "réduit"
    signals=scored["Signaux"] if isinstance(scored["Signaux"],list) else []
    return {"score":float(scored["ScoreRisque"]),"level":scored["NiveauRisque"],"supervised":float(scored["ScoreSupervise"]),"business":float(scored["ScoreMetier"]),"anomaly":float(scored["ScoreAnomalie"]),"signals":signals,"risk_signals":[s for s in signals if s["kind"]=="risk"],"anomaly_signals":[s for s in signals if s["kind"]=="anomaly"],"uncertainty_signals":[s for s in signals if s["kind"]=="uncertainty"],"model_factors":factors}
