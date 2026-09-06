import pandas as pd
from .loader import PRE_CONTROL_COLUMNS, POST_CONTROL_COLUMNS

SCORING_REQUIRED_COLUMNS = [
    "DateCreationSoumissionDossier", "DateOperation", "ReferenceFicheCEE",
    "SecteurOperation", "DomaineOperation", "TypeBeneficiaire", "TypeBatiment",
    "EquipementUtilise", "VolumeCEEDeclare", "Region", "Departement", "Commune",
    "PartenairePseudonymise",
]


def leakage_report(feature_columns: list[str]) -> dict:
    leaked = sorted(set(feature_columns) & set(POST_CONTROL_COLUMNS))
    return {"is_clean": len(leaked) == 0, "leaked_columns": leaked}


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({"column": col, "dtype": str(df[col].dtype), "missing_pct": round(float(df[col].isna().mean() * 100), 2), "n_unique": int(df[col].nunique(dropna=True))})
    return pd.DataFrame(rows).sort_values(["missing_pct", "n_unique"], ascending=[False, False])


def ingestion_report(df: pd.DataFrame) -> dict:
    columns = set(df.columns)
    required = set(SCORING_REQUIRED_COLUMNS)
    expected = set(PRE_CONTROL_COLUMNS)
    post = sorted(columns & set(POST_CONTROL_COLUMNS))
    missing_required = sorted(required - columns)
    missing_optional = sorted((expected - required) - columns)
    extra = sorted(columns - expected - set(POST_CONTROL_COLUMNS))
    key_cols = [c for c in ["ReferenceFicheCEE", "VolumeCEEDeclare", "PartenairePseudonymise"] if c in df.columns]
    rows_with_missing_key = int(df[key_cols].isna().any(axis=1).sum()) if key_cols else len(df)
    invalid_volume = 0
    if "VolumeCEEDeclare" in df.columns:
        vol = pd.to_numeric(df["VolumeCEEDeclare"], errors="coerce")
        invalid_volume = int((vol.isna() | (vol < 0)).sum())
    return {"rows": int(len(df)), "columns": int(len(df.columns)), "missing_required": missing_required, "missing_optional": missing_optional, "post_control_columns": post, "extra_columns": extra, "rows_with_missing_key": rows_with_missing_key, "invalid_volume_rows": invalid_volume, "can_score": len(df) > 0 and not missing_required}


def sanitize_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in POST_CONTROL_COLUMNS if c in df.columns], errors="ignore").copy()
