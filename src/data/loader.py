from pathlib import Path
import pandas as pd
import numpy as np

PRE_CONTROL_COLUMNS = [
    "DateCreationSoumissionDossier", "DateOperation", "ReferenceFicheCEE",
    "TypeOperation", "SecteurOperation", "DomaineOperation", "TypeBeneficiaire",
    "TypeBatiment", "CaracteristiquesTechniques", "SurfaceConcernee",
    "EquipementUtilise", "VolumeCEEDeclare", "VolumeCEECalculeAttribue",
    "Region", "Departement", "Commune", "PartenairePseudonymise",
]

POST_CONTROL_COLUMNS = [
    "DateControle", "ResultatControle", "StatutFinalControle", "MotifNonConformite",
    "TypesAnomalies", "NombreAnomalies", "PresenceDemandeSAV", "MotifDemandeSAV",
    "ResultatApresSAV", "PresenceSecondControle", "ResultatSecondControle", "MotifRejet",
]

POSITIVE_STATUSES = {
    "refusé", "non satisfaisant corrigé", "non satisfaisant",
    "non satisfaisant - en attente de sav", "non satisfaisant - sav effectué",
}
NEGATIVE_STATUSES = {"satisfaisant"}
AMBIGUOUS_STATUSES = {
    "inaccessible / non vérifiable", "partiellement satisfaisant", "non contrôlé",
    "passé sans cofrac", "a effectuer",
}


def load_dataset(path: str | Path, sheet_name=0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError("Format supporté: .xlsx, .xls, .csv")
    return normalize_nulls(df)


def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({"NULL": np.nan, "null": np.nan, "None": np.nan, "": np.nan})
    return out


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    required = set(PRE_CONTROL_COLUMNS)
    if require_target:
        required.add("StatutFinalControle")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes: {missing}")


def build_target(df: pd.DataFrame) -> pd.Series:
    if "StatutFinalControle" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Int64", name="target_risque")
    s = df["StatutFinalControle"].astype("string").str.strip().str.lower()
    y = pd.Series(pd.NA, index=df.index, dtype="Int64", name="target_risque")
    y[s.isin(POSITIVE_STATUSES)] = 1
    y[s.isin(NEGATIVE_STATUSES)] = 0
    return y
