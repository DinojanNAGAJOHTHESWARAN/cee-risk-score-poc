import re
import numpy as np
import pandas as pd

MODEL_NUMERIC_COLUMNS = [
    "VolumeCEEDeclare", "DaysOperationToSubmission", "SubmissionMonth",
    "SubmissionDayOfWeek", "PartnerFrequency", "FicheFrequency", "CommuneFrequency",
    "ComboFrequency", "FicheVolumeRatioToMedian",
]
MODEL_CATEGORICAL_COLUMNS = [
    "ReferenceFicheCEE", "SecteurOperation", "DomaineOperation", "TypeBeneficiaire",
    "TypeBatiment", "Region", "Departement", "EquipmentBrand",
]


def parse_mixed_date(series: pd.Series) -> pd.Series:
    s = series.copy()
    numeric = pd.to_numeric(s, errors="coerce")
    as_text = pd.to_datetime(s.where(numeric.isna()), errors="coerce", dayfirst=False)
    excel_dates = pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric, unit="D")
    return as_text.fillna(excel_dates)


def extract_brand(value):
    if pd.isna(value): return np.nan
    text=str(value)
    m=re.search(r"Marque\s*=\s*([^;|]+)", text, flags=re.I)
    return m.group(1).strip().upper() if m else np.nan


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ["VolumeCEEDeclare", "VolumeCEECalculeAttribue"]:
        x[c] = _safe_num(x.get(c, pd.Series(index=x.index, dtype=float)))
    declared=x["VolumeCEEDeclare"]
    calc=x["VolumeCEECalculeAttribue"]
    x["VolumeGapAbs"]=(declared-calc).abs()
    denom=calc.abs().replace(0,np.nan)
    x["VolumeGapRel"]=(x["VolumeGapAbs"]/denom).clip(upper=20)
    x["VolumeRatio"]=(declared/denom).replace([np.inf,-np.inf],np.nan).clip(upper=20)
    submit=parse_mixed_date(x.get("DateCreationSoumissionDossier", pd.Series(index=x.index,dtype=object)))
    operation=parse_mixed_date(x.get("DateOperation", pd.Series(index=x.index,dtype=object)))
    raw_lag=(submit-operation).dt.total_seconds()/86400
    x["RawDaysOperationToSubmission"]=raw_lag
    x["DaysOperationToSubmission"]=raw_lag.where(raw_lag>=0)
    x["SubmissionYear"]=submit.dt.year
    x["SubmissionMonth"]=submit.dt.month
    x["SubmissionDayOfWeek"]=submit.dt.dayofweek
    x["EquipmentBrand"]=x.get("EquipementUtilise", pd.Series(index=x.index,dtype=object)).map(extract_brand)
    return x


class FeatureContext:
    def __init__(self, alpha: float = 20.0, use_calculated_volume_rules: bool = False):
        self.alpha=float(alpha)
        self.use_calculated_volume_rules=bool(use_calculated_volume_rules)

    def fit(self, history_df: pd.DataFrame, labeled_train_df: pd.DataFrame, y: pd.Series):
        h=base_features(history_df)
        self.partner_freq=h["PartenairePseudonymise"].value_counts(dropna=True).to_dict()
        self.fiche_freq=h["ReferenceFicheCEE"].value_counts(dropna=True).to_dict()
        self.commune_freq=h["Commune"].value_counts(dropna=True).to_dict()
        combo=(h["ReferenceFicheCEE"].astype("string")+"|"+h["TypeBatiment"].astype("string"))
        self.combo_freq=combo.value_counts(dropna=True).to_dict()
        self.fiche_volume_median=h.groupby("ReferenceFicheCEE", dropna=True)["VolumeCEEDeclare"].median().to_dict()
        tmp=labeled_train_df[["PartenairePseudonymise","ReferenceFicheCEE"]].copy()
        tmp["y"]=pd.Series(y, index=labeled_train_df.index).astype(float)
        self.global_rate=float(tmp["y"].mean()) if len(tmp) else 0.10
        self.partner_stats=self._smooth(tmp,"PartenairePseudonymise")
        self.fiche_stats=self._smooth(tmp,"ReferenceFicheCEE")
        return self

    def _smooth(self, df, col):
        g=df.groupby(col, dropna=True)["y"].agg(["count","sum"])
        g["rate"]=(g["sum"]+self.alpha*self.global_rate)/(g["count"]+self.alpha)
        return g[["count","rate"]].to_dict("index")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x=base_features(df)
        x["PartnerFrequency"]=x["PartenairePseudonymise"].map(self.partner_freq).fillna(0).astype(float)
        x["FicheFrequency"]=x["ReferenceFicheCEE"].map(self.fiche_freq).fillna(0).astype(float)
        x["CommuneFrequency"]=x["Commune"].map(self.commune_freq).fillna(0).astype(float)
        combo=x["ReferenceFicheCEE"].astype("string")+"|"+x["TypeBatiment"].astype("string")
        x["ComboFrequency"]=combo.map(self.combo_freq).fillna(0).astype(float)
        med=x["ReferenceFicheCEE"].map(self.fiche_volume_median)
        x["FicheVolumeRatioToMedian"]=(x["VolumeCEEDeclare"]/(pd.to_numeric(med,errors="coerce").replace(0,np.nan))).clip(upper=50)
        for c in MODEL_CATEGORICAL_COLUMNS:
            if c in x.columns:
                x[c] = x[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
        return x

    def partner_risk(self, partner):
        d=self.partner_stats.get(partner)
        return (float(d["rate"]), int(d["count"])) if d else (self.global_rate,0)

    def fiche_risk(self, fiche):
        d=self.fiche_stats.get(fiche)
        return (float(d["rate"]), int(d["count"])) if d else (self.global_rate,0)
