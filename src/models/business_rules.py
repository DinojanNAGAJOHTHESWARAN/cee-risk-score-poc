import pandas as pd

def _signal(kind,title,detail,points=0.0): return {"kind":kind,"title":title,"detail":detail,"points":float(points)}

def business_score_row(row,context):
    score=0.0; signals=[]; gap=row.get("VolumeGapRel")
    if getattr(context,"use_calculated_volume_rules",False):
        if pd.notna(gap):
            if gap>0.15: score+=25; signals.append(_signal("risk","Écart volume déclaré / calculé important",f"L'écart relatif est de {gap:.1%}.",25))
            elif gap>0.05: score+=12; signals.append(_signal("risk","Écart volume déclaré / calculé notable",f"L'écart relatif est de {gap:.1%}.",12))
        elif pd.isna(row.get("VolumeCEECalculeAttribue")): signals.append(_signal("uncertainty","Volume calculé non disponible","Aucun point n'est ajouté pour l'absence seule."))
    pr,count=context.partner_risk(row.get("PartenairePseudonymise"))
    if count>=10:
        if pr>=0.25: score+=25; signals.append(_signal("risk","Historique partenaire défavorable élevé",f"Taux historique lissé : {pr:.1%} sur {count} dossiers qualifiés.",25))
        elif pr>=0.15: score+=15; signals.append(_signal("risk","Historique partenaire défavorable",f"Taux historique lissé : {pr:.1%} sur {count} dossiers qualifiés.",15))
        elif pr>=0.08: score+=8; signals.append(_signal("risk","Historique partenaire au-dessus du niveau bas",f"Taux historique lissé : {pr:.1%} sur {count} dossiers qualifiés.",8))
    else: signals.append(_signal("uncertainty","Historique partenaire insuffisant",f"Seulement {count} dossier(s) qualifié(s). Aucun point automatique."))
    fr,fcount=context.fiche_risk(row.get("ReferenceFicheCEE"))
    if fcount>=20 and fr>=0.20: score+=10; signals.append(_signal("risk","Fiche CEE historiquement plus exposée",f"Taux historique lissé : {fr:.1%} sur {fcount} dossiers qualifiés.",10))
    combo=row.get("ComboFrequency",0)
    if pd.notna(combo) and combo<=2: score+=8; signals.append(_signal("anomaly","Combinaison fiche / bâtiment très rare",f"Cette combinaison apparaît {int(combo)} fois dans l'historique.",8))
    ratio=row.get("FicheVolumeRatioToMedian")
    if pd.notna(ratio):
        if ratio>=4: score+=15; signals.append(_signal("anomaly","Volume très atypique pour la fiche",f"Environ {ratio:.1f} fois la médiane historique.",15))
        elif ratio>=2.5: score+=8; signals.append(_signal("anomaly","Volume supérieur à la référence de la fiche",f"Environ {ratio:.1f} fois la médiane historique.",8))
    lag=row.get("RawDaysOperationToSubmission")
    if pd.notna(lag):
        if lag<0: signals.append(_signal("uncertainty","Chronologie à vérifier",f"La soumission précède la date d'opération de {abs(lag):.0f} jour(s), sans ajout de risque tant que la règle métier n'est pas validée."))
        elif lag>365: score+=8; signals.append(_signal("anomaly","Délai opération → soumission atypique",f"Le délai est de {lag:.0f} jours.",8))
    return min(score,100.0),signals

def business_scores(feature_df,context):
    scores=[]; signals=[]
    for _,row in feature_df.iterrows():
        s,sig=business_score_row(row,context); scores.append(s); signals.append(sig)
    return scores,signals
