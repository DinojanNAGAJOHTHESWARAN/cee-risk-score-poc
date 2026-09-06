# CEE Risk Score — POC

Prototype d’aide à la priorisation des contrôles de dossiers CEE, développé dans le cadre d’un MSc AI Applied to Business.

Le projet combine trois briques : un modèle supervisé, des règles métier explicables et une détection d’anomalies. Le score produit sert à **ordonner une file de contrôle** ; il ne constitue ni une preuve de fraude ni une décision automatique de rejet.

## Confidentialité

Ce dépôt public ne contient **aucune extraction d’entreprise**, aucun modèle entraîné sur ces données et aucun résultat individuel. Les répertoires `data/raw/`, `data/processed/` et les artefacts `models/*.joblib` sont exclus par `.gitignore`. Le seul fichier de données versionné est `data/examples/lot_test_synthetique.csv`, entièrement synthétique et destiné à tester l’interface d’import/scoring.

## Architecture

- `app/streamlit_app.py` : interface Streamlit (tableau de bord, analyse d’un dossier, import & scoring, performance/gouvernance).
- `src/features/` : préparation et feature engineering.
- `src/models/` : modèle supervisé, détection d’anomalies, règles métier et score hybride.
- `src/pipeline/` : entraînement et prédiction.
- `src/data/` : chargement et validation du schéma.
- `config/config.yaml` : paramètres du POC.
- `tests/` : tests unitaires.
- `data/examples/lot_test_synthetique.csv` : lot de test synthétique.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sous Windows :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Entraînement local

Le dépôt ne fournit volontairement pas les données d’entraînement. Placez une extraction autorisée dans `data/raw/` et adaptez `data.default_path` dans `config/config.yaml`, puis lancez :

```bash
python -m src.pipeline.train
```

L’artefact entraîné est créé localement dans `models/` et reste ignoré par Git.

## Lancer l’application

```bash
python -m streamlit run app/streamlit_app.py
```

Une fois un modèle local disponible, l’onglet **Importer & scorer** permet de tester `data/examples/lot_test_synthetique.csv` sans réentraînement.

## Tests

```bash
python -m pytest
```

## Principes de gouvernance

Les variables postérieures au contrôle ne sont pas utilisées comme prédicteurs. La cible supervisée représente un résultat de contrôle défavorable et non une fraude juridiquement confirmée. L’évaluation privilégie ROC-AUC, PR-AUC, rappel, précision et métriques Top-k adaptées à une capacité de contrôle limitée. Le prototype reste human-in-the-loop : le score aide à prioriser, jamais à rejeter automatiquement un dossier.
